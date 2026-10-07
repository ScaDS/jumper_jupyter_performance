/*
 * JUmPER C monitoring collector — main entry point.
 *
 * Handles argument parsing, PID enumeration, process scheduling, the
 * JSON-lines ready handshake, and the main tick loop.  Metric collection
 * is delegated to the per-metric backends registered in g_registry[].
 *
 * Supported levels: process, user, system, slurm.
 *
 * Build:
 *     make -C jumper_extension/monitor/backends/native_c/
 *
 * Usage:
 *     ./jumper_collector --interval 1.0 --target-pid 12345 \
 *                        --levels process,user,system       \
 *                        --collectors cpu,memory,io,gpu
 */

#define _GNU_SOURCE
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
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

#include "monitor.h"
#include "metrics/cpu/cpu.h"
#include "metrics/memory/memory.h"
#include "metrics/io/io.h"
#include "metrics/gpu/gpu.h"

/* ------------------------------------------------------------------ */
/* Global state (extern declarations are in monitor.h)                */
/* ------------------------------------------------------------------ */
static volatile sig_atomic_t g_running      = 1;
static int                   g_target_pid   = -1;
static uid_t                 g_target_uid   = 0;
int                          g_num_cpus     = 0;
int                          g_num_sys_cpus = 0;
long                         g_clk_tck      = 0;
static char g_slurm_job_id[64] = "";

/* Renice: lower target PID tree priority so the collector wins CPU time */
#define RENICE_INCREMENT 19
static int g_reniced_pids[MAX_PIDS];
static int g_reniced_count = 0;

/* Active levels and active collectors */
static int g_level_active[MAX_LEVELS];
static const char *g_level_names[] = {
    "process", "user", "system", "slurm"
};

static const CCollector * const g_registry[] = {
    &cpu_collector,
    &memory_collector,
    &io_collector,
    &gpu_collector,
};
#define N_COLLECTORS (int)(sizeof(g_registry) / sizeof(g_registry[0]))
static int g_collector_active[N_COLLECTORS];

/* ------------------------------------------------------------------ */
/* Shared utilities (declared in collector.h)                         */
/* ------------------------------------------------------------------ */

int read_file(const char *path, char *buf, size_t bufsz) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    size_t n = fread(buf, 1, bufsz - 1, f);
    buf[n] = '\0';
    fclose(f);
    return (int)n;
}

void emit_per_device_agg(FILE *fp, const char *prefix,
                         const double *vals, int n) {
    if (n <= 0) return;
    double avg = 0.0, mn = vals[0], mx = vals[0];
    for (int i = 0; i < n; i++) {
        avg += vals[i];
        if (vals[i] < mn) mn = vals[i];
        if (vals[i] > mx) mx = vals[i];
    }
    avg /= n;
    fprintf(fp, ",\"%savg\":%.6f,\"%smin\":%.6f,\"%smax\":%.6f",
            prefix, avg, prefix, mn, prefix, mx);
    for (int i = 0; i < n; i++)
        fprintf(fp, ",\"%s%d\":%.6f", prefix, i, vals[i]);
}

/* ------------------------------------------------------------------ */
/* Timing helpers                                                     */
/* ------------------------------------------------------------------ */

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

/* ------------------------------------------------------------------ */
/* Signal handling                                                    */
/* ------------------------------------------------------------------ */

static void sig_handler(int sig) { (void)sig; g_running = 0; }

/* ------------------------------------------------------------------ */
/* Target PID renice                                                  */
/* ------------------------------------------------------------------ */

static void renice_target_pids(int *pids, int npids) {
    pid_t my_pid = getpid();
    struct sched_param sp = { .sched_priority = 0 };
    for (int i = 0; i < npids; i++) {
        if (pids[i] == my_pid) continue;
        int already = 0;
        for (int j = 0; j < g_reniced_count; j++)
            if (g_reniced_pids[j] == pids[i]) { already = 1; break; }
        if (already) continue;
        if (setpriority(PRIO_PROCESS, pids[i], RENICE_INCREMENT) == 0) {
            sched_setscheduler(pids[i], SCHED_BATCH, &sp);
            if (g_reniced_count < MAX_PIDS)
                g_reniced_pids[g_reniced_count++] = pids[i];
        }
    }
}

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

static int collect_pid_tree(int root, int *out, int max_out) {
    int count = 0;
    out[count++] = root;
    int head = 0;
    while (head < count && count < max_out) {
        char path[128], buf[8192];
        int parent = out[head++];
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

static int collect_uid_pids(uid_t uid, int *out, int max_out) {
    int count = 0;
    DIR *d = opendir("/proc");
    if (!d) return 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && count < max_out) {
        if (!isdigit((unsigned char)ent->d_name[0])) continue;
        int pid = atoi(ent->d_name);
        char path[64], buf[2048];
        snprintf(path, sizeof(path), "/proc/%d/status", pid);
        if (read_file(path, buf, sizeof(buf)) < 0) continue;
        char *line = strstr(buf, "\nUid:");
        if (!line) continue;
        if ((uid_t)strtoul(line + 5, NULL, 10) == uid)
            out[count++] = pid;
    }
    closedir(d);
    return count;
}

static int collect_slurm_pids(int *out, int max_out) {
    if (g_slurm_job_id[0] == '\0') return 0;
    /* Seed with the target process's own PID tree so the slurm level
       always has at least the monitored process's PIDs, even when
       /proc/<pid>/environ is inaccessible due to ptrace restrictions. */
    int count = collect_pid_tree(g_target_pid, out, max_out);
    char needle[128];
    int needle_len = snprintf(needle, sizeof(needle),
                              "SLURM_JOB_ID=%s", g_slurm_job_id);
    DIR *d = opendir("/proc");
    if (!d) return 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL && count < max_out) {
        if (!isdigit((unsigned char)ent->d_name[0])) continue;
        int pid = atoi(ent->d_name);
        char path[64], buf[32768];
        snprintf(path, sizeof(path), "/proc/%d/environ", pid);
        int n = read_file(path, buf, sizeof(buf));
        if (n <= 0) continue;
        int found = 0;
        for (int off = 0; off < n && !found; ) {
            int elen = (int)strnlen(buf + off, n - off);
            if (elen >= needle_len &&
                memcmp(buf + off, needle, needle_len) == 0 &&
                (buf[off + needle_len] == '\0' ||
                 buf[off + needle_len] == '\n'))
                found = 1;
            off += elen + 1;
        }
        if (found) {
            /* Skip duplicates already in the array from the PID tree seed */
            int dup = 0;
            for (int j = 0; j < count; j++) {
                if (out[j] == pid) { dup = 1; break; }
            }
            if (!dup) out[count++] = pid;
        }
    }
    closedir(d);
    return count;
}

/* ------------------------------------------------------------------ */
/* Argument parsing                                                   */
/* ------------------------------------------------------------------ */

static void parse_levels(const char *arg) {
    char tmp[256];
    strncpy(tmp, arg, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    char *tok = strtok(tmp, ",");
    while (tok) {
        for (int i = 0; i < MAX_LEVELS; i++)
            if (strcmp(tok, g_level_names[i]) == 0) g_level_active[i] = 1;
        tok = strtok(NULL, ",");
    }
}

static void parse_collectors(const char *arg) {
    char tmp[256];
    strncpy(tmp, arg, sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';
    char *tok = strtok(tmp, ",");
    while (tok) {
        for (int i = 0; i < N_COLLECTORS; i++)
            if (strcmp(tok, g_registry[i]->name) == 0) g_collector_active[i] = 1;
        tok = strtok(NULL, ",");
    }
}

/* ------------------------------------------------------------------ */
/* Ready handshake                                                    */
/* ------------------------------------------------------------------ */

static void emit_ready(void) {
    /* Detect CPU count from target process affinity */
    cpu_set_t cpuset;
    if (sched_getaffinity(g_target_pid, sizeof(cpuset), &cpuset) == 0)
        g_num_cpus = CPU_COUNT(&cpuset);
    else
        g_num_cpus = get_nprocs();
    g_num_sys_cpus = get_nprocs();

    /* Build cpu_handles list */
    int handles[MAX_CPUS], nhandles = 0;
    if (sched_getaffinity(g_target_pid, sizeof(cpuset), &cpuset) == 0)
        for (int i = 0; i < g_num_sys_cpus && nhandles < MAX_CPUS; i++)
            if (CPU_ISSET(i, &cpuset)) handles[nhandles++] = i;

    /* Memory limits */
    long rlim_bytes = -1;
    {
        struct rlimit rl;
        if (getrlimit(RLIMIT_AS, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY)
            rlim_bytes = (long)rl.rlim_cur;
    }
    double sys_mem = 0.0;
    {
        char buf[4096];
        if (read_file("/proc/meminfo", buf, sizeof(buf)) > 0) {
            char *p = strstr(buf, "MemTotal:");
            if (p) sys_mem = strtol(p + 9, NULL, 10) / (1024.0 * 1024.0);
        }
    }

    int ngpus = gpu_num_gpus();

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
        ngpus, gpu_memory_gb(), gpu_name());

    /* SLURM_MEM_PER_NODE fallback (in MB → GB) */
    double slurm_mem = -1.0;
    {
        const char *env_mem = getenv("SLURM_MEM_PER_NODE");
        if (env_mem && env_mem[0])
            slurm_mem = atof(env_mem) / 1024.0;
    }

    int first_ml = 1;
    for (int i = 0; i < MAX_LEVELS; i++) {
        if (!g_level_active[i]) continue;
        if (!first_ml) fputc(',', stdout);
        first_ml = 0;
        if (i == LEVEL_PROCESS && rlim_bytes > 0)
            fprintf(stdout, "\"%s\":%.2f", g_level_names[i],
                    (double)rlim_bytes / (1024.0 * 1024.0 * 1024.0));
        else if (i == LEVEL_SLURM && slurm_mem > 0.0)
            fprintf(stdout, "\"%s\":%.2f", g_level_names[i], slurm_mem);
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

    fprintf(stdout, "],\"columns_by_level\":{");
    {
        int first = 1;
        for (int i = 0; i < MAX_LEVELS; i++) {
            if (!g_level_active[i]) continue;
            if (!first) fputc(',', stdout);
            first = 0;
            int ncpus = (i == LEVEL_SYSTEM) ? g_num_sys_cpus : g_num_cpus;
            fprintf(stdout, "\"%s\":[\"time\"", g_level_names[i]);
            for (int c = 0; c < N_COLLECTORS; c++)
                if (g_collector_active[c])
                    g_registry[c]->emit_columns(stdout, i, ncpus, ngpus);
            fputc(']', stdout);
        }
    }
    fprintf(stdout, "}}\n");
    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* Main tick                                                          */
/* ------------------------------------------------------------------ */

static void emit_tick(double perf_time, double dt) {
    static TickContext ctx;
    ctx.n_proc = ctx.n_user = ctx.n_slurm = ctx.n_all = 0;

    if (g_level_active[LEVEL_PROCESS])
        ctx.n_proc = collect_pid_tree(g_target_pid, ctx.pids_proc, MAX_PIDS);
    if (g_level_active[LEVEL_USER])
        ctx.n_user = collect_uid_pids(g_target_uid, ctx.pids_user, MAX_PIDS);
    if (g_level_active[LEVEL_SLURM])
        ctx.n_slurm = collect_slurm_pids(ctx.pids_slurm, MAX_PIDS);

    /* Build all_pids union for snapshot calls */
    for (int i = 0; i < ctx.n_proc  && ctx.n_all < MAX_PIDS; i++)
        ctx.all_pids[ctx.n_all++] = ctx.pids_proc[i];
    for (int i = 0; i < ctx.n_user  && ctx.n_all < MAX_PIDS; i++)
        ctx.all_pids[ctx.n_all++] = ctx.pids_user[i];
    for (int i = 0; i < ctx.n_slurm && ctx.n_all < MAX_PIDS; i++)
        ctx.all_pids[ctx.n_all++] = ctx.pids_slurm[i];

    renice_target_pids(ctx.all_pids, ctx.n_all);

    for (int c = 0; c < N_COLLECTORS; c++)
        if (g_collector_active[c])
            g_registry[c]->snapshot(&ctx);

    double wallclock = wall_sec();

    for (int lv = 0; lv < MAX_LEVELS; lv++) {
        if (!g_level_active[lv]) continue;
        fprintf(stdout, "{\"wallclock\":%.6f,\"level\":\"%s\","
                "\"sample\":{\"time\":%.6f",
                wallclock, g_level_names[lv], perf_time);
        for (int c = 0; c < N_COLLECTORS; c++)
            if (g_collector_active[c])
                g_registry[c]->emit_sample(stdout, lv, &ctx, dt);
        fprintf(stdout, "}}\n");
    }

    for (int c = 0; c < N_COLLECTORS; c++)
        if (g_collector_active[c] && g_registry[c]->post_tick)
            g_registry[c]->post_tick();

    fflush(stdout);
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    double      interval       = 1.0;
    const char *levels_str     = NULL;
    const char *collectors_str = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--interval") == 0 && i + 1 < argc)
            interval = atof(argv[++i]);
        else if (strcmp(argv[i], "--target-pid") == 0 && i + 1 < argc)
            g_target_pid = atoi(argv[++i]);
        else if (strcmp(argv[i], "--levels") == 0 && i + 1 < argc)
            levels_str = argv[++i];
        else if (strcmp(argv[i], "--collectors") == 0 && i + 1 < argc)
            collectors_str = argv[++i];
    }

    if (g_target_pid <= 0) g_target_pid = getppid();

    /* Elevate scheduling priority; silently ignore permission errors */
    if (setpriority(PRIO_PROCESS, 0, -10) != 0 && errno != EACCES) {}

    g_clk_tck    = sysconf(_SC_CLK_TCK);
    g_target_uid = getuid();
    {
        char path[64], buf[2048];
        snprintf(path, sizeof(path), "/proc/%d/status", g_target_pid);
        if (read_file(path, buf, sizeof(buf)) > 0) {
            char *p = strstr(buf, "\nUid:");
            if (p) g_target_uid = (uid_t)strtoul(p + 5, NULL, 10);
        }
    }

    /* Detect SLURM_JOB_ID from own environment or target process environ */
    {
        const char *env_jid = getenv("SLURM_JOB_ID");
        if (env_jid && env_jid[0]) {
            snprintf(g_slurm_job_id, sizeof(g_slurm_job_id), "%s", env_jid);
        } else {
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

    /* Active levels */
    if (levels_str) {
        parse_levels(levels_str);
    } else {
        g_level_active[LEVEL_PROCESS] = 1;
        g_level_active[LEVEL_USER]    = 1;
        g_level_active[LEVEL_SYSTEM]  = 1;
        if (g_slurm_job_id[0] != '\0') g_level_active[LEVEL_SLURM] = 1;
    }

    /* Active collectors */
    if (collectors_str) {
        parse_collectors(collectors_str);
    } else {
        for (int c = 0; c < N_COLLECTORS; c++) g_collector_active[c] = 1;
    }

    signal(SIGTERM, sig_handler);
    signal(SIGINT,  sig_handler);
    signal(SIGPIPE, SIG_IGN);

    /* Setup phase — GPU init happens here */
    for (int c = 0; c < N_COLLECTORS; c++)
        if (g_collector_active[c]) g_registry[c]->setup();

    emit_ready();

    /* Prime CPU tick cache so the first sample has a valid delta */
    {
        int tmp_pids[MAX_PIDS];
        int n = collect_pid_tree(g_target_pid, tmp_pids, MAX_PIDS);
        cpu_prime(tmp_pids, n, g_num_sys_cpus);
    }

    /* Main loop */
    double next_tick = monotonic_sec();
    double prev_tick = next_tick;

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
        if (dt <= 0) dt = interval;

        emit_tick(now, dt);
    }

    restore_target_pids();

    for (int c = 0; c < N_COLLECTORS; c++)
        if (g_collector_active[c] && g_registry[c]->teardown)
            g_registry[c]->teardown();

    return 0;
}
