#!/bin/sh
set -eu
case "$(uname -s)" in
    Darwin) KOKORO_RUNTIME=metal exec ./ci.sh ;;
    *) echo "Kokoro's GPU example exercises the Metal runtime on Apple platforms" ;;
esac
