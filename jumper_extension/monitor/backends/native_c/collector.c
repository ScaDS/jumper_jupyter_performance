/*
 * JUmPER C monitoring collector
 *
 * Lightweight alternative to the Python collector.  Reads metrics directly
 * from /proc and writes the same JSON-lines protocol to stdout so that
 * SubprocessPerformanceMonitor can consume the stream unchanged.
 *
 * Supported levels: "process", "user", "system", "slurm".
 * GPU metrics are collected via NVML (libnvidia-ml.so) if available,
 * loaded dynamically at runtime — no compile-time dependency.
 *
 * Build:
 *     gcc -O2 -o jumper_collector collector.c -lm -ldl
 *
 * Usage:
 *     ./jumper_collector --interval 1.0 --target-pid 12345 \
 *                        --levels process,user,system
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Tunables                                                           */
/* ------------------------------------------------------------------ */
#define MAX_PIDS      4096
#define MAX_CPUS       512
#define MAX_LEVELS       4
#define LEVEL_PROCESS    0
#define LEVEL_USER       1
#define LEVEL_SYSTEM     2
#define LEVEL_SLURM      3

/* ------------------------------------------------------------------ */
/* Global state                                                       */
/* ------------------------------------------------------------------ */
static volatile sig_atomic_t g_running = 1;
static int   g_target_pid    = -1;     /* PID tree root to observe     */
static uid_t g_target_uid    = 0;      /* UID for "user" level         */
static int   g_num_cpus      = 0;      /* CPUs available to target     */
static int   g_num_sys_cpus  = 0;      /* total online CPUs            */
static long  g_clk_tck       = 0;      /* sysconf(_SC_CLK_TCK)         */
static char  g_slurm_job_id[64] = ""; /* SLURM_JOB_ID for slurm level */

/* Renice: lower target PID tree priority so the collector wins CPU time */
#define RENICE_INCREMENT  19
static int g_reniced_pids[MAX_PIDS];
static int g_reniced_count = 0;

/* Active levels (bitmap) */
static int g_level_active[MAX_LEVELS];
static const char *g_level_names[] = {
    "process", "user", "system", "slurm"
};
/* (g_n_levels removed — count derived from g_level_active) */

/* ------------------------------------------------------------------ */
/* Per-PID previous CPU ticks (for delta-based cpu_percent)            */
/* ------------------------------------------------------------------ */
typedef struct {
    int    pid;
    long   prev_utime;
    long   prev_stime;
    int    valid;        /* had a previous sample */
} pid_cpu_t;

static pid_cpu_t g_pid_cpu[MAX_PIDS];
static int       g_pid_cpu_count = 0;

/* Per-CPU previous ticks (for per-core system-level utilisation) */
static long g_prev_cpu_total[MAX_CPUS];
static long g_prev_cpu_idle[MAX_CPUS];
static int  g_prev_cpu_valid = 0;

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */

static void sig_handler(int sig) {
    (void)sig;
    g_running = 0;
}

static double monotonic_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static double wall_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Read a whole small file into buf.  Returns bytes read or -1. */
static int read_file(const char *path, char *buf, size_t bufsz) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, bufsz - 1, f);
    buf[n] = '\0';
    fclose(f);
    return (int)n;
}

/* ------------------------------------------------------------------ */
/* Target PID renice                                                  */
/* ------------------------------------------------------------------ */

/* Set nice of target PIDs to +RENICE_INCREMENT (skip our own PID).
   The target root PID (IPython kernel) is included so that all future
   children it spawns inherit the elevated nice value automatically.
   Silently ignores EPERM / ESRCH.  Already-reniced PIDs are skipped. */
static void renice_target_pids(int *pids, int npids) {
    pid_t my_pid = getpid();
    struct sched_param sp = { .sched_priority = 0 };
    for (int i = 0; i < npids; i++) {
        if (pids[i] == my_pid) continue;
        /* check if already reniced */
        int already = 0;
        for (int j = 0; j < g_reniced_count; j++) {
            if (g_reniced_pids[j] == pids[i]) { already = 1; break; }
        }
        if (already) continue;
        if (setpriority(PRIO_PROCESS, pids[i], RENICE_INCREMENT) == 0) {
            /* Also set SCHED_BATCH so CFS treats them as throughput-
               oriented and avoids preempting the collector. */
            sched_setscheduler(pids[i], SCHED_BATCH, &sp);
            if (g_reniced_count < MAX_PIDS)
                g_reniced_pids[g_reniced_count++] = pids[i];
        }
    }
}

/* Restore all reniced PIDs back to nice 0 / SCHED_OTHER. */
static void restore_target_pids(void) {
    struct sched_param sp = { .sched_priority = 0 };
    for (int i = 0; i < g_reniced_count; i++) {
        setpriority(PRIO_PROCESS, g_reniced_pids[i], 0);
        sched_setscheduler(g_reniced_pids[i], SCHED_OTHER, &sp);
    }
    g_reniced_count = 0;
}

/* ------------------------------------------------------------------ */
/* PID enumeration                                                    */
/* ------------------------------------------------------------------ */

/* Collect target PID + all recursive children.  Returns count.       */
static int collect_pid_tree(int root, int *out, int max_out) {
    int count = 0;
    /* start with root */
    out[count++] = root;

    /* BFS over /proc/[pid]/task/[tid]/children or fall back to
       scanning /proc/[pid]/stat ppid.  The children file is fastest. */
    int head = 0;
    while (head < count && count < max_out) {
        char path[128];
        char buf[8192];
        int parent = out[head++];

        /* Try /proc/<pid>/task/<pid>/children first (Linux ≥3.5) */
        snprintf(path, sizeof(path),
                 "/proc/%d/task/%d/children", parent, parent);
        int n = read_file(path, buf, sizeof(buf));
        if (n > 0) {
            char *p = buf;
            while (*p && count < max_out) {
                while (*p == ' ') p++;
                if (!*p) break;
                int cpid = (int)strtol(p, &p, 10);
                if (cpid > 0) out[count++] = cpid;
            }
        }
    }
    return count;
}

/* Collect all PIDs owned by uid.  Returns count. */
static int collect_uid_pids(uid_t uid, int *out, int max_out) {
    int count = 0;
    DIR *d = opendir("/proc");
    if (!d) return 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && count < max_out) {
        if (!isdigit((unsigned char)ent->d_name[0])) continue;
        int pid = atoi(ent->d_name);
        char path[64];
        snprintf(path, sizeof(path), "/proc/%d/status", pid);
        char buf[2048];
        if (read_file(path, buf, sizeof(buf)) < 0) continue;
        /* Find "Uid:\t<real>\t..." */
        char *line = strstr(buf, "\nUid:");
        if (!line) continue;
        uid_t ruid = (uid_t)strtoul(line + 5, NULL, 10);
        if (ruid == uid) out[count++] = pid;
    }
    closedir(d);
    return count;
}

/* Collect PIDs whose environ contains SLURM_JOB_ID=<g_slurm_job_id>. */
static int collect_slurm_pids(int *out, int max_out) {
    if (g_slurm_job_id[0] == '\0') return 0;
    int count = 0;
    char needle[128];
    int needle_len = snprintf(needle, sizeof(needle),
                              "SLURM_JOB_ID=%s", g_slurm_job_id);
    DIR *d = opendir("/proc");
    if (!d) return 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && count < max_out) {
        if (!isdigit((unsigned char)ent->d_name[0])) continue;
        int pid = atoi(ent->d_name);
        char path[64];
        snprintf(path, sizeof(path), "/proc/%d/environ", pid);
        /* environ can be large; read a reasonable chunk */
        char buf[32768];
        int n = read_file(path, buf, sizeof(buf));
        if (n <= 0) continue;
        /* environ is NUL-separated — search for needle within [0..n) */
        int found = 0;
        for (int off = 0; off < n && !found; ) {
            int elen = (int)strnlen(buf + off, n - off);
            if (elen >= needle_len &&
                memcmp(buf + off, needle, needle_len) == 0 &&
                (buf[off + needle_len] == '\0' ||
                 buf[off + needle_len] == '\n')) {
                found = 1;
            }
            off += elen + 1;
        }
        if (found) out[count++] = pid;
    }
    closedir(d);
    return count;
}

/* ------------------------------------------------------------------ */
/* CPU utilisation                                                    */
/* ------------------------------------------------------------------ */

/* Read utime + stime from /proc/<pid>/stat.  Returns 0 on success.  */
static int read_pid_cpu(int pid, long *utime, long *stime) {
    char path[64], buf[1024];
    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    if (read_file(path, buf, sizeof(buf)) < 0) return -1;
    /* Fields after the comm (which may contain spaces/parens):
       skip to the closing ')' then parse from field 3 onwards.       */
    char *p = strrchr(buf, ')');
    if (!p) return -1;
    p += 2; /* skip ") " */
    /* Fields: state(3) ppid(4) ... utime(14) stime(15)              */
    long vals[16];
    int  idx = 3;
    char *tok = strtok(p, " ");
    while (tok && idx <= 15) {
        vals[idx] = strtol(tok, NULL, 10);
        idx++;
        tok = strtok(NULL, " ");
    }
    if (idx <= 15) return -1;
    *utime = vals[14];
    *stime = vals[15];
    return 0;
}

/* Look up or create the cache entry for pid. */
static pid_cpu_t *get_pid_cpu(int pid) {
    for (int i = 0; i < g_pid_cpu_count; i++) {
        if (g_pid_cpu[i].pid == pid) return &g_pid_cpu[i];
    }
    if (g_pid_cpu_count >= MAX_PIDS) return NULL;
    pid_cpu_t *e = &g_pid_cpu[g_pid_cpu_count++];
    e->pid = pid;
    e->prev_utime = 0;
    e->prev_stime = 0;
    e->valid = 0;
    return e;
}

/* Per-tick snapshot of current (utime, stime) for every PID.
   Built once per tick so that multiple levels can compute deltas
   from the same baseline without double-consuming the cache. */
typedef struct { int pid; long utime, stime; } cpu_snap_t;
static cpu_snap_t g_cpu_snap[MAX_PIDS];
static int        g_cpu_snap_count = 0;

/* Read current CPU ticks for all unique PIDs across all active levels.
   Call once per tick, before any compute_pid_set_cpu calls. */
static void snapshot_cpu_ticks(int *all_pids, int n_all) {
    g_cpu_snap_count = 0;
    for (int i = 0; i < n_all && g_cpu_snap_count < MAX_PIDS; i++) {
        /* deduplicate */
        int dup = 0;
        for (int j = 0; j < g_cpu_snap_count; j++) {
            if (g_cpu_snap[j].pid == all_pids[i]) { dup = 1; break; }
        }
        if (dup) continue;
        long ut = 0, st = 0;
        if (read_pid_cpu(all_pids[i], &ut, &st) < 0) continue;
        cpu_snap_t *s = &g_cpu_snap[g_cpu_snap_count++];
        s->pid = all_pids[i];
        s->utime = ut;
        s->stime = st;
    }
}

/* Look up snapshot entry for pid. */
static cpu_snap_t *find_snap(int pid) {
    for (int i = 0; i < g_cpu_snap_count; i++) {
        if (g_cpu_snap[i].pid == pid) return &g_cpu_snap[i];
    }
    return NULL;
}

/* Compute total CPU% for a set of PIDs using snapshot + cache.
   Does NOT update the cache — call commit_pid_cpu_cache afterwards. */
static double compute_pid_set_cpu(int *pids, int npids, double dt_sec) {
    double total = 0.0;
    for (int i = 0; i < npids; i++) {
        cpu_snap_t *s = find_snap(pids[i]);
        if (!s) continue;
        pid_cpu_t *e = get_pid_cpu(pids[i]);
        if (!e) continue;
        if (e->valid) {
            long d = (s->utime - e->prev_utime) + (s->stime - e->prev_stime);
            double pct = (double)d / (g_clk_tck * dt_sec) * 100.0;
            total += pct;
        }
    }
    return total;
}

/* Commit snapshot values into the cache.  Call once per tick after
   all levels have been computed. */
static void commit_pid_cpu_cache(void) {
    for (int i = 0; i < g_cpu_snap_count; i++) {
        pid_cpu_t *e = get_pid_cpu(g_cpu_snap[i].pid);
        if (!e) continue;
        e->prev_utime = g_cpu_snap[i].utime;
        e->prev_stime = g_cpu_snap[i].stime;
        e->valid = 1;
    }
}

/* Remove cache entries not present in the snapshot (dead PIDs). */
static void prune_pid_cpu_cache(void) {
    for (int i = 0; i < g_pid_cpu_count; i++) {
        int found = 0;
        for (int j = 0; j < g_cpu_snap_count; j++) {
            if (g_pid_cpu[i].pid == g_cpu_snap[j].pid) { found = 1; break; }
        }
        if (!found) {
            g_pid_cpu[i] = g_pid_cpu[--g_pid_cpu_count];
            i--;
        }
    }
}

/* Per-core system CPU.  Fills util_pct[0..ncpus-1].  Returns 0 on ok. */
static int read_system_cpu_per_core(double *util_pct, int ncpus) {
    char buf[16384];
    if (read_file("/proc/stat", buf, sizeof(buf)) < 0) return -1;

    /* Skip the aggregate "cpu ..." line */
    char *line = buf;
    int cpu_idx = 0;
    while ((line = strchr(line, '\n')) != NULL && cpu_idx < ncpus) {
        line++; /* skip newline */
        if (strncmp(line, "cpu", 3) != 0 || !isdigit((unsigned char)line[3]))
            continue;
        /* parse: cpuN user nice system idle iowait irq softirq steal */
        long vals[8];
        char *p = line + 3;
        while (isdigit((unsigned char)*p)) p++; /* skip cpu index */
        for (int i = 0; i < 8; i++) {
            vals[i] = strtol(p, &p, 10);
        }
        long total = 0;
        for (int i = 0; i < 8; i++) total += vals[i];
        long idle = vals[3] + vals[4]; /* idle + iowait */

        if (g_prev_cpu_valid && cpu_idx < MAX_CPUS) {
            long dt = total - g_prev_cpu_total[cpu_idx];
            long di = idle  - g_prev_cpu_idle[cpu_idx];
            util_pct[cpu_idx] = dt > 0
                ? (double)(dt - di) / (double)dt * 100.0
                : 0.0;
        } else {
            util_pct[cpu_idx] = 0.0;
        }
        if (cpu_idx < MAX_CPUS) {
            g_prev_cpu_total[cpu_idx] = total;
            g_prev_cpu_idle[cpu_idx]  = idle;
        }
        cpu_idx++;
    }
    g_prev_cpu_valid = 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Memory (RSS via /proc/<pid>/statm — fast, single-line read)        */
/* ------------------------------------------------------------------ */

static long read_pid_rss_kb(int pid) {
    char path[64], buf[128];
    snprintf(path, sizeof(path), "/proc/%d/statm", pid);
    if (read_file(path, buf, sizeof(buf)) < 0) return 0;
    /* statm fields: size resident shared text lib data dt (in pages) */
    long pages;
    long dummy;
    if (sscanf(buf, "%ld %ld", &dummy, &pages) != 2) return 0;
    /* Convert pages to kB (page size is almost always 4 kB) */
    return pages * (sysconf(_SC_PAGESIZE) / 1024);
}

/* Per-tick memory snapshot — read once, look up per level. */
typedef struct { int pid; long rss_kb; } mem_snap_t;
static mem_snap_t g_mem_snap[MAX_PIDS];
static int        g_mem_snap_count = 0;

static long find_mem_snap(int pid) {
    for (int i = 0; i < g_mem_snap_count; i++)
        if (g_mem_snap[i].pid == pid) return g_mem_snap[i].rss_kb;
    return 0;
}

static double compute_pid_set_memory_gb(int *pids, int npids) {
    long total_kb = 0;
    for (int i = 0; i < npids; i++)
        total_kb += find_mem_snap(pids[i]);
    return (double)total_kb / (1024.0 * 1024.0);
}

/* System-level: total - available */
static double system_memory_used_gb(void) {
    char buf[4096];
    if (read_file("/proc/meminfo", buf, sizeof(buf)) < 0) return 0.0;
    long total = 0, avail = 0;
    char *p = strstr(buf, "MemTotal:");
    if (p) total = strtol(p + 9, NULL, 10);
    p = strstr(buf, "MemAvailable:");
    if (p) avail = strtol(p + 13, NULL, 10);
    return (double)(total - avail) / (1024.0 * 1024.0);
}

/* ------------------------------------------------------------------ */
/* I/O counters                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    long read_count;
    long write_count;
    long read_bytes;
    long write_bytes;
} io_counters_t;

static io_counters_t read_pid_io(int pid) {
    io_counters_t c = {0, 0, 0, 0};
    char path[64], buf[1024];
    snprintf(path, sizeof(path), "/proc/%d/io", pid);
    if (read_file(path, buf, sizeof(buf)) < 0) return c;
    char *p;
    if ((p = strstr(buf, "syscr:")))       c.read_count  = strtol(p + 6, NULL, 10);
    if ((p = strstr(buf, "syscw:")))       c.write_count = strtol(p + 6, NULL, 10);
    if ((p = strstr(buf, "read_bytes:")))  c.read_bytes  = strtol(p + 11, NULL, 10);
    if ((p = strstr(buf, "write_bytes:"))) c.write_bytes = strtol(p + 12, NULL, 10);
    return c;
}

/* Per-tick IO snapshot — read once, look up per level. */
typedef struct { int pid; io_counters_t io; } io_snap_t;
static io_snap_t g_io_snap[MAX_PIDS];
static int       g_io_snap_count = 0;

static io_counters_t find_io_snap(int pid) {
    for (int i = 0; i < g_io_snap_count; i++)
        if (g_io_snap[i].pid == pid) return g_io_snap[i].io;
    return (io_counters_t){0, 0, 0, 0};
}

/* Read memory + IO for all unique PIDs once per tick.
   Call after snapshot_cpu_ticks (same all_pids union). */
static void snapshot_mem_io(int *all_pids, int n_all) {
    g_mem_snap_count = 0;
    g_io_snap_count  = 0;
    for (int i = 0; i < n_all; i++) {
        /* deduplicate */
        int dup = 0;
        for (int j = 0; j < g_mem_snap_count; j++) {
            if (g_mem_snap[j].pid == all_pids[i]) { dup = 1; break; }
        }
        if (dup) continue;
        if (g_mem_snap_count < MAX_PIDS) {
            g_mem_snap[g_mem_snap_count].pid    = all_pids[i];
            g_mem_snap[g_mem_snap_count].rss_kb = read_pid_rss_kb(all_pids[i]);
            g_mem_snap_count++;
        }
        if (g_io_snap_count < MAX_PIDS) {
            g_io_snap[g_io_snap_count].pid = all_pids[i];
            g_io_snap[g_io_snap_count].io  = read_pid_io(all_pids[i]);
            g_io_snap_count++;
        }
    }
}

static io_counters_t compute_pid_set_io(int *pids, int npids) {
    io_counters_t total = {0, 0, 0, 0};
    for (int i = 0; i < npids; i++) {
        io_counters_t c = find_io_snap(pids[i]);
        total.read_count  += c.read_count;
        total.write_count += c.write_count;
        total.read_bytes  += c.read_bytes;
        total.write_bytes += c.write_bytes;
    }
    return total;
}

/* Read system-wide disk IO from /proc/diskstats.                     *
 * Matches psutil.disk_io_counters() behaviour: sums all devices.     *
 * Fields per line (kernel doc):                                      *
 *   major minor name reads_completed _ sectors_read _ _ writes_completed _ sectors_written ... *
 * We use: field 4 = reads_completed, field 8 = writes_completed,     *
 *         field 6 = sectors_read, field 10 = sectors_written.        *
 * Sector size is assumed 512 bytes (standard Linux ABI).             */
static io_counters_t read_system_disk_io(void) {
    io_counters_t total = {0, 0, 0, 0};
    FILE *f = fopen("/proc/diskstats", "r");
    if (!f) return total;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        unsigned int major, minor;
        char devname[128];
        long f1, f2, f3, f4, f5, f6, f7, f8;
        int n = sscanf(line,
            " %u %u %127s %ld %ld %ld %ld %ld %ld %ld %ld",
            &major, &minor, devname,
            &f1, &f2, &f3, &f4,   /* reads_completed, reads_merged, sectors_read, time_reading */
            &f5, &f6, &f7, &f8);  /* writes_completed, writes_merged, sectors_written, time_writing */
        if (n < 11) continue;
        /* Skip partitions: include only whole-disk devices.
           Simple heuristic: skip if name ends with a digit preceded by
           a letter (e.g. sda1, nvme0n1p1). Include sda, nvme0n1, etc. */
        int len = (int)strlen(devname);
        if (len > 0 && devname[len-1] >= '0' && devname[len-1] <= '9') {
            /* Check if it looks like a partition number */
            int j = len - 1;
            while (j > 0 && devname[j] >= '0' && devname[j] <= '9') j--;
            if (j > 0 && devname[j] == 'p' && j > 1) {
                continue;  /* e.g. nvme0n1p1 */
            }
            if (j > 0 && ((devname[j] >= 'a' && devname[j] <= 'z') ||
                          (devname[j] >= 'A' && devname[j] <= 'Z'))) {
                /* Could be sda1 — only skip if the letter is not 'n'
                   followed by a digit (nvme0n1 pattern) */
                if (!(devname[j] == 'n' && j > 0 &&
                      devname[j-1] >= '0' && devname[j-1] <= '9')) {
                    continue;  /* e.g. sda1, vdb2 */
                }
            }
        }
        total.read_count  += f1;          /* reads_completed */
        total.write_count += f5;          /* writes_completed */
        total.read_bytes  += f3 * 512L;   /* sectors_read * 512 */
        total.write_bytes += f7 * 512L;   /* sectors_written * 512 */
    }
    fclose(f);
    return total;
}

/* ------------------------------------------------------------------ */
/* GPU via NVML (dynamic loading)                                     */
/* ------------------------------------------------------------------ */

#define NVML_SUCCESS            0
#define NVML_MAX_GPUS          16
#define NVML_MAX_PROCS        128
#define NVML_DEVICE_NAME_LEN  128

typedef int   nvmlReturn_t;
typedef void *nvmlDevice_t;

typedef struct { unsigned long long total, free, used; } nvmlMemory_t;
typedef struct { unsigned int gpu, memory;             } nvmlUtilization_t;
typedef struct { unsigned int pid; unsigned long long usedGpuMemory; } nvmlProcessInfo_t;

/* Function pointer types */
typedef nvmlReturn_t (*fn_nvmlInit)(void);
typedef nvmlReturn_t (*fn_nvmlShutdown)(void);
typedef nvmlReturn_t (*fn_nvmlDeviceGetCount)(unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
typedef nvmlReturn_t (*fn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetComputeRunningProcesses)(nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);

static struct {
    void *handle;                   /* dlopen handle                   */
    int   available;                /* 1 if NVML loaded & initialised  */
    int   num_gpus;
    nvmlDevice_t devices[NVML_MAX_GPUS];
    double       gpu_memory_gb;     /* total memory of first GPU       */
    char         gpu_name[256];

    fn_nvmlInit                            Init;
    fn_nvmlShutdown                        Shutdown;
    fn_nvmlDeviceGetCount                  GetCount;
    fn_nvmlDeviceGetHandleByIndex          GetHandle;
    fn_nvmlDeviceGetName                   GetName;
    fn_nvmlDeviceGetMemoryInfo             GetMemInfo;
    fn_nvmlDeviceGetUtilizationRates       GetUtil;
    fn_nvmlDeviceGetComputeRunningProcesses GetProcs;
} g_nvml = {0};

static void nvml_init(void) {
    g_nvml.handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!g_nvml.handle)
        g_nvml.handle = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!g_nvml.handle) return;

    /* Resolve symbols */
    g_nvml.Init     = (fn_nvmlInit)dlsym(g_nvml.handle, "nvmlInit_v2");
    g_nvml.Shutdown = (fn_nvmlShutdown)dlsym(g_nvml.handle, "nvmlShutdown");
    g_nvml.GetCount = (fn_nvmlDeviceGetCount)dlsym(g_nvml.handle, "nvmlDeviceGetCount_v2");
    g_nvml.GetHandle= (fn_nvmlDeviceGetHandleByIndex)dlsym(g_nvml.handle, "nvmlDeviceGetHandleByIndex_v2");
    g_nvml.GetName  = (fn_nvmlDeviceGetName)dlsym(g_nvml.handle, "nvmlDeviceGetName");
    g_nvml.GetMemInfo=(fn_nvmlDeviceGetMemoryInfo)dlsym(g_nvml.handle, "nvmlDeviceGetMemoryInfo");
    g_nvml.GetUtil  = (fn_nvmlDeviceGetUtilizationRates)dlsym(g_nvml.handle, "nvmlDeviceGetUtilizationRates");
    g_nvml.GetProcs = (fn_nvmlDeviceGetComputeRunningProcesses)dlsym(g_nvml.handle, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (!g_nvml.GetProcs)
        g_nvml.GetProcs = (fn_nvmlDeviceGetComputeRunningProcesses)dlsym(g_nvml.handle, "nvmlDeviceGetComputeRunningProcesses");

    if (!g_nvml.Init || !g_nvml.GetCount || !g_nvml.GetHandle ||
        !g_nvml.GetMemInfo || !g_nvml.GetUtil) {
        dlclose(g_nvml.handle);
        g_nvml.handle = NULL;
        return;
    }

    if (g_nvml.Init() != NVML_SUCCESS) {
        dlclose(g_nvml.handle);
        g_nvml.handle = NULL;
        return;
    }

    unsigned int cnt = 0;
    if (g_nvml.GetCount(&cnt) != NVML_SUCCESS || cnt == 0) {
        if (g_nvml.Shutdown) g_nvml.Shutdown();
        dlclose(g_nvml.handle);
        g_nvml.handle = NULL;
        return;
    }
    if ((int)cnt > NVML_MAX_GPUS) cnt = NVML_MAX_GPUS;
    g_nvml.num_gpus = (int)cnt;

    for (unsigned int i = 0; i < cnt; i++) {
        g_nvml.GetHandle(i, &g_nvml.devices[i]);
    }

    /* GPU name + memory from first device */
    if (g_nvml.GetName) {
        g_nvml.GetName(g_nvml.devices[0], g_nvml.gpu_name,
                       sizeof(g_nvml.gpu_name));
    }
    nvmlMemory_t mi;
    if (g_nvml.GetMemInfo(g_nvml.devices[0], &mi) == NVML_SUCCESS) {
        g_nvml.gpu_memory_gb = (double)mi.total / (1024.0*1024.0*1024.0);
    }

    g_nvml.available = 1;
}

static void nvml_shutdown(void) {
    if (g_nvml.handle) {
        if (g_nvml.Shutdown) g_nvml.Shutdown();
        dlclose(g_nvml.handle);
        g_nvml.handle = NULL;
    }
}

/* Collect GPU metrics for one device. */
typedef struct {
    double util;   /* GPU utilisation %  */
    double band;   /* bandwidth (always 0 for now) */
    double mem_gb; /* memory used in GiB */
} gpu_sample_t;

/* System-level: full device stats */
static gpu_sample_t nvml_collect_system(int dev_idx) {
    gpu_sample_t s = {0, 0, 0};
    if (!g_nvml.available || dev_idx >= g_nvml.num_gpus) return s;
    nvmlUtilization_t u;
    if (g_nvml.GetUtil(g_nvml.devices[dev_idx], &u) == NVML_SUCCESS)
        s.util = (double)u.gpu;
    nvmlMemory_t mi;
    if (g_nvml.GetMemInfo(g_nvml.devices[dev_idx], &mi) == NVML_SUCCESS)
        s.mem_gb = (double)mi.used / (1024.0*1024.0*1024.0);
    return s;
}

/* Process-level: memory used by target PID set; util attributed if mem>0 */
static gpu_sample_t nvml_collect_process(int dev_idx,
                                         int *pids, int npids) {
    gpu_sample_t s = {0, 0, 0};
    if (!g_nvml.available || !g_nvml.GetProcs ||
        dev_idx >= g_nvml.num_gpus) return s;

    nvmlProcessInfo_t procs[NVML_MAX_PROCS];
    unsigned int count = NVML_MAX_PROCS;
    if (g_nvml.GetProcs(g_nvml.devices[dev_idx], &count, procs) != NVML_SUCCESS)
        return s;

    unsigned long long proc_mem = 0;
    for (unsigned int i = 0; i < count; i++) {
        for (int j = 0; j < npids; j++) {
            if ((int)procs[i].pid == pids[j] && procs[i].usedGpuMemory) {
                proc_mem += procs[i].usedGpuMemory;
                break;
            }
        }
    }
    s.mem_gb = (double)proc_mem / (1024.0*1024.0*1024.0);
    if (s.mem_gb > 0) {
        nvmlUtilization_t u;
        if (g_nvml.GetUtil(g_nvml.devices[dev_idx], &u) == NVML_SUCCESS)
            s.util = (double)u.gpu;
    }
    return s;
}

/* ------------------------------------------------------------------ */
/* JSON output helpers                                                */
/* ------------------------------------------------------------------ */

/* Write a JSON array of doubles. */
static void json_double_array(FILE *f, double *arr, int n) {
    fputc('[', f);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', f);
        if (isnan(arr[i]) || isinf(arr[i]))
            fprintf(f, "0.0");
        else
            fprintf(f, "%.4f", arr[i]);
    }
    fputc(']', f);
}

/* ------------------------------------------------------------------ */
/* Handshake                                                          */
/* ------------------------------------------------------------------ */

static void emit_ready(void) {
    /* Detect CPU count via sched_getaffinity of target */
    cpu_set_t cpuset;
    if (sched_getaffinity(g_target_pid, sizeof(cpuset), &cpuset) == 0) {
        g_num_cpus = CPU_COUNT(&cpuset);
    } else {
        g_num_cpus = get_nprocs();
    }
    g_num_sys_cpus = get_nprocs();

    /* Build cpu_handles list */
    int handles[MAX_CPUS];
    int nhandles = 0;
    if (sched_getaffinity(g_target_pid, sizeof(cpuset), &cpuset) == 0) {
        for (int i = 0; i < g_num_sys_cpus && nhandles < MAX_CPUS; i++) {
            if (CPU_ISSET(i, &cpuset)) handles[nhandles++] = i;
        }
    }

    /* Detect memory limits */
    long rlim_bytes = -1;
    {
        struct rlimit rl;
        if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
            rlim_bytes = (long)rl.rlim_cur;
        }
    }
    double sys_mem = 0.0;
    {
        char buf[4096];
        if (read_file("/proc/meminfo", buf, sizeof(buf)) > 0) {
            char *p = strstr(buf, "MemTotal:");
            if (p) sys_mem = strtol(p + 9, NULL, 10) / (1024.0 * 1024.0);
        }
    }

    /* Build levels list */
    fprintf(stdout,
        "{\"status\":\"ready\","
        "\"pid\":%d,"
        "\"num_cpus\":%d,"
        "\"num_system_cpus\":%d,"
        "\"num_gpus\":%d,"
        "\"gpu_memory\":%.2f,"
        "\"gpu_name\":\"%s\","
        "\"memory_limits\":{",
        getpid(), g_num_cpus, g_num_sys_cpus,
        g_nvml.available ? g_nvml.num_gpus : 0,
        g_nvml.available ? g_nvml.gpu_memory_gb : 0.0,
        g_nvml.available ? g_nvml.gpu_name : "");

    int first_ml = 1;
    for (int i = 0; i < MAX_LEVELS; i++) {
        if (!g_level_active[i]) continue;
        if (!first_ml) fputc(',', stdout);
        first_ml = 0;
        if (i == LEVEL_PROCESS && rlim_bytes > 0)
            fprintf(stdout, "\"%s\":%.2f", g_level_names[i],
                    (double)rlim_bytes / (1024.0*1024.0*1024.0));
        else
            fprintf(stdout, "\"%s\":%.2f", g_level_names[i], sys_mem);
    }

    fprintf(stdout, "},\"cpu_handles\":[");
    for (int i = 0; i < nhandles; i++) {
        if (i) fputc(',', stdout);
        fprintf(stdout, "%d", handles[i]);
    }

    fprintf(stdout, "],\"levels\":[");
    {
        int first = 1;
        for (int i = 0; i < MAX_LEVELS; i++) {
            if (!g_level_active[i]) continue;
            if (!first) fputc(',', stdout);
            first = 0;
            fprintf(stdout, "\"%s\"", g_level_names[i]);
        }
    }
    fprintf(stdout, "]}\n");
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Emit one sample per active level                                   */
/* ------------------------------------------------------------------ */

static void emit_samples(double perf_time, double dt) {
    static int pids_proc[MAX_PIDS];
    static int pids_user[MAX_PIDS];
    static int pids_slurm[MAX_PIDS];
    int n_proc = 0, n_user = 0, n_slurm = 0;

    /* Collect PID sets only once per tick */
    if (g_level_active[LEVEL_PROCESS])
        n_proc = collect_pid_tree(g_target_pid, pids_proc, MAX_PIDS);
    if (g_level_active[LEVEL_USER])
        n_user = collect_uid_pids(g_target_uid, pids_user, MAX_PIDS);
    if (g_level_active[LEVEL_SLURM])
        n_slurm = collect_slurm_pids(pids_slurm, MAX_PIDS);

    /* Build union of all PIDs and snapshot their CPU ticks once.
       This ensures multiple levels can compute deltas from the same
       baseline without double-consuming cached values. */
    {
        static int all_pids[MAX_PIDS];
        int n_all = 0;
        for (int i = 0; i < n_proc && n_all < MAX_PIDS; i++)
            all_pids[n_all++] = pids_proc[i];
        for (int i = 0; i < n_user && n_all < MAX_PIDS; i++)
            all_pids[n_all++] = pids_user[i];
        for (int i = 0; i < n_slurm && n_all < MAX_PIDS; i++)
            all_pids[n_all++] = pids_slurm[i];

        /* Lower the priority of the target PID tree so the
           collector (at nice 0) is scheduled preferentially. */
        renice_target_pids(all_pids, n_all);

        snapshot_cpu_ticks(all_pids, n_all);
        snapshot_mem_io(all_pids, n_all);
    }

    double wallclock = wall_sec();

    for (int lv = 0; lv < MAX_LEVELS; lv++) {
        if (!g_level_active[lv]) continue;

        double cpu_arr[MAX_CPUS];
        double memory;
        io_counters_t io;
        int ncpus_out;

        /* Determine which PID set to use */
        int *pids_set = NULL;
        int  npids_set = 0;

        if (lv == LEVEL_SYSTEM) {
            ncpus_out = g_num_sys_cpus;
            read_system_cpu_per_core(cpu_arr, ncpus_out);
            memory = system_memory_used_gb();
            io = read_system_disk_io();
        } else if (lv == LEVEL_PROCESS) {
            pids_set = pids_proc; npids_set = n_proc;
            ncpus_out = g_num_cpus;
            double total_pct = compute_pid_set_cpu(pids_set, npids_set, dt);
            double per_core = total_pct / g_num_cpus;
            for (int i = 0; i < ncpus_out; i++) cpu_arr[i] = per_core;
            memory = compute_pid_set_memory_gb(pids_set, npids_set);
            io = compute_pid_set_io(pids_set, npids_set);
        } else if (lv == LEVEL_USER) {
            pids_set = pids_user; npids_set = n_user;
            ncpus_out = g_num_cpus;
            double total_pct = compute_pid_set_cpu(pids_set, npids_set, dt);
            double per_core = total_pct / g_num_cpus;
            for (int i = 0; i < ncpus_out; i++) cpu_arr[i] = per_core;
            memory = compute_pid_set_memory_gb(pids_set, npids_set);
            io = compute_pid_set_io(pids_set, npids_set);
        } else { /* LEVEL_SLURM */
            pids_set = pids_slurm; npids_set = n_slurm;
            ncpus_out = g_num_cpus;
            double total_pct = compute_pid_set_cpu(pids_set, npids_set, dt);
            double per_core = total_pct / g_num_cpus;
            for (int i = 0; i < ncpus_out; i++) cpu_arr[i] = per_core;
            memory = compute_pid_set_memory_gb(pids_set, npids_set);
            io = compute_pid_set_io(pids_set, npids_set);
        }

        /* GPU metrics */
        double gpu_util[NVML_MAX_GPUS];
        double gpu_band[NVML_MAX_GPUS];
        double gpu_mem[NVML_MAX_GPUS];
        int ngpus = g_nvml.available ? g_nvml.num_gpus : 0;

        for (int g = 0; g < ngpus; g++) {
            gpu_sample_t gs;
            if (lv == LEVEL_SYSTEM) {
                gs = nvml_collect_system(g);
            } else {
                /* process / user / slurm — attribute to PID set */
                gs = nvml_collect_process(g, pids_set, npids_set);
            }
            gpu_util[g] = gs.util;
            gpu_band[g] = gs.band;
            gpu_mem[g]  = gs.mem_gb;
        }

        fprintf(stdout, "{\"time\":%.6f,\"wallclock\":%.6f,\"level\":\"%s\","
                "\"sample\":{\"cpu_util\":",
                perf_time, wallclock, g_level_names[lv]);
        json_double_array(stdout, cpu_arr, ncpus_out);
        fprintf(stdout, ",\"memory\":%.6f,\"gpu_util\":", memory);
        json_double_array(stdout, gpu_util, ngpus);
        fprintf(stdout, ",\"gpu_band\":");
        json_double_array(stdout, gpu_band, ngpus);
        fprintf(stdout, ",\"gpu_mem\":");
        json_double_array(stdout, gpu_mem, ngpus);
        fprintf(stdout, ",\"io_counters\":[%ld,%ld,%ld,%ld]}}\n",
                io.read_count, io.write_count,
                io.read_bytes, io.write_bytes);
    }

    /* Commit this tick's snapshot into the cache and prune dead PIDs */
    commit_pid_cpu_cache();
    prune_pid_cpu_cache();

    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

static void parse_levels(const char *arg) {
    /* comma-separated: process,user,system,slurm */
    char tmp[256];
    strncpy(tmp, arg, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    char *tok = strtok(tmp, ",");
    while (tok) {
        for (int i = 0; i < MAX_LEVELS; i++) {
            if (strcmp(tok, g_level_names[i]) == 0)
                g_level_active[i] = 1;
        }
        tok = strtok(NULL, ",");
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    double interval = 1.0;
    const char *levels_str = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc)
            interval = atof(argv[++i]);
        else if (strcmp(argv[i], "--target-pid") == 0 && i + 1 < argc)
            g_target_pid = atoi(argv[++i]);
        else if (strcmp(argv[i], "--levels") == 0 && i + 1 < argc)
            levels_str = argv[++i];
    }

    if (g_target_pid <= 0) {
        g_target_pid = getppid();
    }

    /* Elevate scheduling priority so the collector is not starved when
       all CPU cores are saturated by compute workloads.  A negative nice
       value requires CAP_SYS_NICE or root; silently ignore EACCES. */
    if (setpriority(PRIO_PROCESS, 0, -10) != 0 && errno != EACCES) {
        /* non-permission error — just continue at default priority */
    }

    g_clk_tck = sysconf(_SC_CLK_TCK);
    g_target_uid = getuid();
    /* Try to read the UID of the target process */
    {
        char path[64], buf[2048];
        snprintf(path, sizeof(path), "/proc/%d/status", g_target_pid);
        if (read_file(path, buf, sizeof(buf)) > 0) {
            char *p = strstr(buf, "\nUid:");
            if (p) g_target_uid = (uid_t)strtoul(p + 5, NULL, 10);
        }
    }

    /* Detect SLURM_JOB_ID from environment (inherit from parent or
       read from target process's /proc/<pid>/environ). */
    {
        const char *env_jid = getenv("SLURM_JOB_ID");
        if (env_jid && env_jid[0]) {
            snprintf(g_slurm_job_id, sizeof(g_slurm_job_id), "%s", env_jid);
        } else {
            /* Read from target process environ */
            char epath[64], ebuf[32768];
            snprintf(epath, sizeof(epath), "/proc/%d/environ", g_target_pid);
            int en = read_file(epath, ebuf, sizeof(ebuf));
            if (en > 0) {
                for (int off = 0; off < en; ) {
                    int elen = (int)strnlen(ebuf + off, en - off);
                    if (strncmp(ebuf + off, "SLURM_JOB_ID=", 13) == 0) {
                        snprintf(g_slurm_job_id, sizeof(g_slurm_job_id),
                                 "%s", ebuf + off + 13);
                        break;
                    }
                    off += elen + 1;
                }
            }
        }
    }

    /* Default levels: process, user, system (+ slurm if available) */
    if (levels_str) {
        parse_levels(levels_str);
    } else {
        g_level_active[LEVEL_PROCESS] = 1;
        g_level_active[LEVEL_USER]    = 1;
        g_level_active[LEVEL_SYSTEM]  = 1;
        if (g_slurm_job_id[0] != '\0')
            g_level_active[LEVEL_SLURM] = 1;
    }

    signal(SIGTERM, sig_handler);
    signal(SIGINT,  sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* ---- GPU init (dynamic, no compile-time dependency) ---- */
    nvml_init();

    /* ---- handshake ---- */
    emit_ready();

    /* ---- main loop ---- */
    double next_tick = monotonic_sec();
    double prev_tick = next_tick;

    /* prime CPU counters with a dummy read */
    {
        int tmp_pids[MAX_PIDS];
        int n = collect_pid_tree(g_target_pid, tmp_pids, MAX_PIDS);
        for (int i = 0; i < n; i++) {
            long ut = 0, st = 0;
            if (read_pid_cpu(tmp_pids[i], &ut, &st) == 0) {
                pid_cpu_t *e = get_pid_cpu(tmp_pids[i]);
                if (e) {
                    e->prev_utime = ut;
                    e->prev_stime = st;
                    e->valid = 1;
                }
            }
        }
        double dummy_arr[MAX_CPUS];
        read_system_cpu_per_core(dummy_arr, g_num_sys_cpus);
    }

    while (g_running) {
        next_tick += interval;
        double delay = next_tick - monotonic_sec();
        if (delay > 0) {
            struct timespec ts;
            ts.tv_sec  = (time_t)delay;
            ts.tv_nsec = (long)((delay - (double)ts.tv_sec) * 1e9);
            nanosleep(&ts, NULL);
        } else {
            next_tick = monotonic_sec();
        }

        double now = monotonic_sec();
        double dt  = now - prev_tick;
        prev_tick  = now;

        if (dt <= 0) dt = interval; /* guard */

        emit_samples(now, dt);
    }

    restore_target_pids();
    nvml_shutdown();
    return 0;
}
