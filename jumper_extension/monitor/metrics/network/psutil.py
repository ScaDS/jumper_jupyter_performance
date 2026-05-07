from __future__ import annotations

import psutil

from jumper_extension.monitor.metrics.context import CollectionContext
from jumper_extension.monitor.metrics.network.common import NetworkCollectorBackend


class PsutilNetworkCollector(NetworkCollectorBackend):
    """Network I/O backend implemented via psutil.

    Returns system-wide byte and packet counters for all interfaces combined.
    Per-process network statistics are not available via psutil on Linux without
    elevated privileges, so process/user/slurm levels report zeros.
    """

    name = "network-psutil"

    def collect(self, level: str, context: CollectionContext) -> list[int]:
        if level == "system":
            net = psutil.net_io_counters()
            if net:
                return [net.bytes_sent, net.bytes_recv,
                        net.packets_sent, net.packets_recv]
        return [0, 0, 0, 0]
