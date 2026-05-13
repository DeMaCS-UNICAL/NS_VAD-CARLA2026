# DP-SR runtime

This directory contains the  DP-SR runtime used by the Python agent.

Expected layout:

```text
DP-sr.jar
reasoner/
reasoner/lib/
```

`DP-sr.jar` must be executable with `java -jar`. The Python wrapper starts the
Java process from this directory so DP-SR can find its bundled `reasoner/`
support files.

If DP-SR is updated, replace the files here and update this file with the source
URL, version, date, and license details.
