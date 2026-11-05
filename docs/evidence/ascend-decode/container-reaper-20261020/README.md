# Containers on 187 are destroyed within minutes

2026-10-20.

    1792506655 exec_start: bash /tmp/wait187.sh trt-asc
    1792506794 kill trt-asc
    1792506794 exec_die trt-asc
    1792506795 die trt-asc
    1792506795 destroy trt-asc

The arm started at 22:29:29 and the container was killed and destroyed at
22:33:14. Over three hours the event stream records four destroys: `trt-asc`
twice, and the other tenant's `505bvl_omni_c_c0` and `505bvl_omni_d_d0` once
each. So this is not aimed at unregistered containers or at me.

No cron entry, systemd timer or host process on 187 matches; a3-84's container
has run for a day. Whatever does it is external to the host and specific to
this one.

## Why it looked like four different problems

A destroyed container leaves nothing for `docker ps -a` to show and no exit
code to inspect, so each occurrence presented as the next plausible thing:

- a3-84, exit 137 mid-load -- read as the NFS prefetch storm, since vLLM turns
  prefetching on by itself when it sees an NFS checkpoint that fits in 90% of
  RAM. That mechanism is real and the `safetensors_load_strategy="lazy"` fix
  is right, but it was not why the container went away.
- 194 -- read as host memory exhaustion. dmesg's OOM entries were from
  September.
- 187, first time -- correctly mine: a diagnostic command that also ran
  `mkcontainer.sh`, whose first line is `docker rm -f trt-asc`, while the arm
  was running.
- 187, second time -- nothing of mine, which is what finally sent me to
  `docker events`.

## What it settles

Whole-machine arms go on a3-84, whose container survives. 187 is for single-die
probes: the launch floor, the routing costs, the group list, the absorbed
gather and scatter, and mixbrick's self-check all ran there today inside a
minute each and none was touched.

The cost is that a3-84's dies are the contended ones. So the two constraints
are now explicit and opposite -- a3-84 has a durable container and busy dies,
187 has free dies and a container that does not last an arm.
