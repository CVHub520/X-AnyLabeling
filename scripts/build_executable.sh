#!/bin/bash
system=$1

if [ $system = "win-cpu" ]; then
    echo "Building Windows CPU version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm x-anylabeling-win-cpu.spec
elif [ $system = "win-gpu" ];then
    echo "Building Windows GPU version..."
    export X_ANYLABELING_DEVICE=GPU
    pyinstaller --noconfirm x-anylabeling-win-gpu.spec
elif [ $system = "linux-cpu" ];then
    echo "Building Linux CPU version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm x-anylabeling-linux-cpu.spec
elif [ $system = "linux-gpu" ];then
    echo "Building Linux GPU version..."
    export X_ANYLABELING_DEVICE=GPU
    pyinstaller --noconfirm x-anylabeling-linux-gpu.spec
elif [ $system = "macos" ];then
    echo "Building macOS version..."
    export X_ANYLABELING_DEVICE=CPU
    pyinstaller --noconfirm x-anylabeling-macos.spec
else
    echo "System value '$system' is not recognized."
fi