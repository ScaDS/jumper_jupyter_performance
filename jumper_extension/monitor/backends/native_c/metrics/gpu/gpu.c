/*
 * GPU metric backend — NVIDIA NVML via dynamic loading.
 *
 * libnvidia-ml.so is loaded at runtime so no compile-time dependency
 * on NVIDIA headers or libraries is required.  On machines without a
 * GPU the setup() call is a silent no-op and all emit functions output
 * nothing (ngpus == 0).
 */

#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

#include "gpu.h"

/* ------------------------------------------------------------------ */
/* NVML type and constant definitions                                 */
/* ------------------------------------------------------------------ */

#define NVML_SUCCESS   0
#define NVML_MAX_GPUS 16
#define NVML_MAX_PROCS 128

typedef int   nvmlReturn_t;
typedef void *nvmlDevice_t;

typedef struct { unsigned long long total, free, used; } nvmlMemory_t;
typedef struct { unsigned int gpu, memory;             } nvmlUtilization_t;
typedef struct {
    unsigned int pid;
    unsigned long long usedGpuMemory;
} nvmlProcessInfo_t;

typedef nvmlReturn_t (*fn_nvmlInit)(void);
typedef nvmlReturn_t (*fn_nvmlShutdown)(void);
typedef nvmlReturn_t (*fn_nvmlDeviceGetCount)(unsigned int *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetHandleByIndex)(unsigned int, nvmlDevice_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetName)(nvmlDevice_t, char *, unsigned int);
typedef nvmlReturn_t (*fn_nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetUtilizationRates)(nvmlDevice_t, nvmlUtilization_t *);
typedef nvmlReturn_t (*fn_nvmlDeviceGetComputeRunningProcesses)(
    nvmlDevice_t, unsigned int *, nvmlProcessInfo_t *);

/* ------------------------------------------------------------------ */
/* Private state                                                      */
/* ------------------------------------------------------------------ */

static struct {
    void        *handle;
    int          available;
    int          num_gpus;
    nvmlDevice_t devices[NVML_MAX_GPUS];
    double       gpu_memory_gb;
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

/* ------------------------------------------------------------------ */
/* Public accessors (used by emit_ready() in collector_main.c)        */
/* ------------------------------------------------------------------ */

int         gpu_num_gpus(void)  { return g_nvml.available ? g_nvml.num_gpus      : 0;    }
double      gpu_memory_gb(void) { return g_nvml.available ? g_nvml.gpu_memory_gb : 0.0;  }
const char *gpu_name(void)      { return g_nvml.available ? g_nvml.gpu_name      : "";   }

/* ------------------------------------------------------------------ */
/* Private helpers                                                    */
/* ------------------------------------------------------------------ */

typedef struct { double util, band, mem_gb; } gpu_sample_t;

static gpu_sample_t collect_system(int dev_idx) {
    gpu_sample_t s = {0, 0, 0};
    if (!g_nvml.available || dev_idx >= g_nvml.num_gpus) return s;
    nvmlUtilization_t u;
    if (g_nvml.GetUtil(g_nvml.devices[dev_idx], &u) == NVML_SUCCESS) {
        s.util = (double)u.gpu;
        s.band = (double)u.memory;
    }
    nvmlMemory_t mi;
    if (g_nvml.GetMemInfo(g_nvml.devices[dev_idx], &mi) == NVML_SUCCESS)
        s.mem_gb = (double)mi.used / (1024.0 * 1024.0 * 1024.0);
    return s;
}

static gpu_sample_t collect_process(int dev_idx, int *pids, int npids) {
    gpu_sample_t s = {0, 0, 0};
    if (!g_nvml.available || !g_nvml.GetProcs || dev_idx >= g_nvml.num_gpus)
        return s;
    nvmlProcessInfo_t procs[NVML_MAX_PROCS];
    unsigned int count = NVML_MAX_PROCS;
    if (g_nvml.GetProcs(g_nvml.devices[dev_idx], &count, procs) != NVML_SUCCESS)
        return s;
    unsigned long long proc_mem = 0;
    for (unsigned int i = 0; i < count; i++)
        for (int j = 0; j < npids; j++)
            if ((int)procs[i].pid == pids[j] && procs[i].usedGpuMemory) {
                proc_mem += procs[i].usedGpuMemory;
                break;
            }
    s.mem_gb = (double)proc_mem / (1024.0 * 1024.0 * 1024.0);
    if (s.mem_gb > 0) {
        nvmlUtilization_t u;
        if (g_nvml.GetUtil(g_nvml.devices[dev_idx], &u) == NVML_SUCCESS) {
            s.util = (double)u.gpu;
            s.band = (double)u.memory;
        }
    }
    return s;
}

/* ------------------------------------------------------------------ */
/* CCollector implementation                                          */
/* ------------------------------------------------------------------ */

static void gpu_setup(void) {
    g_nvml.handle = dlopen("libnvidia-ml.so.1", RTLD_NOW);
    if (!g_nvml.handle)
        g_nvml.handle = dlopen("libnvidia-ml.so", RTLD_NOW);
    if (!g_nvml.handle) return;

    g_nvml.Init     = (fn_nvmlInit)dlsym(g_nvml.handle, "nvmlInit_v2");
    g_nvml.Shutdown = (fn_nvmlShutdown)dlsym(g_nvml.handle, "nvmlShutdown");
    g_nvml.GetCount = (fn_nvmlDeviceGetCount)dlsym(g_nvml.handle, "nvmlDeviceGetCount_v2");
    g_nvml.GetHandle= (fn_nvmlDeviceGetHandleByIndex)dlsym(g_nvml.handle, "nvmlDeviceGetHandleByIndex_v2");
    g_nvml.GetName  = (fn_nvmlDeviceGetName)dlsym(g_nvml.handle, "nvmlDeviceGetName");
    g_nvml.GetMemInfo=(fn_nvmlDeviceGetMemoryInfo)dlsym(g_nvml.handle, "nvmlDeviceGetMemoryInfo");
    g_nvml.GetUtil  = (fn_nvmlDeviceGetUtilizationRates)dlsym(g_nvml.handle, "nvmlDeviceGetUtilizationRates");
    g_nvml.GetProcs = (fn_nvmlDeviceGetComputeRunningProcesses)dlsym(
        g_nvml.handle, "nvmlDeviceGetComputeRunningProcesses_v3");
    if (!g_nvml.GetProcs)
        g_nvml.GetProcs = (fn_nvmlDeviceGetComputeRunningProcesses)dlsym(
            g_nvml.handle, "nvmlDeviceGetComputeRunningProcesses");

    if (!g_nvml.Init || !g_nvml.GetCount || !g_nvml.GetHandle ||
        !g_nvml.GetMemInfo || !g_nvml.GetUtil) {
        dlclose(g_nvml.handle); g_nvml.handle = NULL; return;
    }
    if (g_nvml.Init() != NVML_SUCCESS) {
        dlclose(g_nvml.handle); g_nvml.handle = NULL; return;
    }

    unsigned int cnt = 0;
    if (g_nvml.GetCount(&cnt) != NVML_SUCCESS || cnt == 0) {
        if (g_nvml.Shutdown) g_nvml.Shutdown();
        dlclose(g_nvml.handle); g_nvml.handle = NULL; return;
    }
    if ((int)cnt > NVML_MAX_GPUS) cnt = NVML_MAX_GPUS;
    g_nvml.num_gpus = (int)cnt;
    for (unsigned int i = 0; i < cnt; i++)
        g_nvml.GetHandle(i, &g_nvml.devices[i]);

    if (g_nvml.GetName)
        g_nvml.GetName(g_nvml.devices[0], g_nvml.gpu_name, sizeof(g_nvml.gpu_name));
    nvmlMemory_t mi;
    if (g_nvml.GetMemInfo(g_nvml.devices[0], &mi) == NVML_SUCCESS)
        g_nvml.gpu_memory_gb = (double)mi.total / (1024.0 * 1024.0 * 1024.0);

    g_nvml.available = 1;
}

static void gpu_snapshot(TickContext *ctx) { (void)ctx; }

static void gpu_emit_columns(FILE *fp, int level_idx, int n_cpus, int n_gpus) {
    (void)level_idx; (void)n_cpus;
    if (n_gpus <= 0) return;
    const char *metrics[] = {"util", "band", "mem"};
    for (int m = 0; m < 3; m++) {
        fprintf(fp, ",\"gpu_%s_avg\",\"gpu_%s_min\",\"gpu_%s_max\"",
                metrics[m], metrics[m], metrics[m]);
        for (int g = 0; g < n_gpus; g++)
            fprintf(fp, ",\"gpu_%s_%d\"", metrics[m], g);
    }
}

static void gpu_emit_sample(FILE *fp, int level_idx, TickContext *ctx, double dt) {
    (void)dt;
    int ngpus = g_nvml.available ? g_nvml.num_gpus : 0;
    if (ngpus <= 0) return;

    double gpu_util[NVML_MAX_GPUS], gpu_band[NVML_MAX_GPUS], gpu_mem[NVML_MAX_GPUS];
    int *pids = NULL, npids = 0;
    if (level_idx != LEVEL_SYSTEM) {
        switch (level_idx) {
            case LEVEL_PROCESS: pids = ctx->pids_proc;  npids = ctx->n_proc;  break;
            case LEVEL_USER:    pids = ctx->pids_user;  npids = ctx->n_user;  break;
            default:            pids = ctx->pids_slurm; npids = ctx->n_slurm; break;
        }
    }
    for (int g = 0; g < ngpus; g++) {
        gpu_sample_t gs = (level_idx == LEVEL_SYSTEM)
            ? collect_system(g) : collect_process(g, pids, npids);
        gpu_util[g] = gs.util;
        gpu_band[g] = gs.band;
        gpu_mem[g]  = gs.mem_gb;
    }
    emit_per_device_agg(fp, "gpu_util_", gpu_util, ngpus);
    emit_per_device_agg(fp, "gpu_band_", gpu_band, ngpus);
    emit_per_device_agg(fp, "gpu_mem_",  gpu_mem,  ngpus);
}

static void gpu_teardown(void) {
    if (g_nvml.handle) {
        if (g_nvml.Shutdown) g_nvml.Shutdown();
        dlclose(g_nvml.handle);
        g_nvml.handle = NULL;
    }
}

const CCollector gpu_collector = {
    .name         = "gpu",
    .setup        = gpu_setup,
    .snapshot     = gpu_snapshot,
    .emit_columns = gpu_emit_columns,
    .emit_sample  = gpu_emit_sample,
    .post_tick    = NULL,
    .teardown     = gpu_teardown,
};
