# Benchmarking Troubleshooting Guide

This guide covers common benchmarking issues and provides step-by-step diagnostics using CLI flags and OS tools.

## Table of Contents
1. [Benchmark Flakiness](#benchmark-flakiness)
2. [High Variance in Results](#high-variance-in-results)
3. [Memory Leaks](#memory-leaks)
4. [CPU Throttling](#cpu-throttling)
5. [System Resource Contention](#system-resource-contention)
6. [Garbage Collection Issues](#garbage-collection-issues)

---

## Benchmark Flakiness

### Symptoms
- Inconsistent benchmark results across runs
- Random failures or timeouts
- Results that vary significantly between identical runs

### Step-by-Step Diagnostics

#### 1. Check System Load
```bash
# Monitor current system load
uptime

# Check CPU usage and processes
htop

# Monitor system activity
vmstat 1 10
```

#### 2. Verify CPU Frequency Scaling
```bash
# Check current CPU governor
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Check available governors
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors

# Check current CPU frequencies
cat /proc/cpuinfo | grep "cpu MHz"
```

#### 3. Monitor Network Interference (for network benchmarks)
```bash
# Check network interface statistics
cat /proc/net/dev

# Monitor network traffic
netstat -i

# Check for network errors
ethtool -S eth0 | grep -i error
```

#### 4. Run Benchmark with Isolation
```bash
# Set CPU affinity to specific cores
taskset -c 0-3 ./your_benchmark

# Run with high priority
nice -n -10 ./your_benchmark

# Disable CPU frequency scaling temporarily
sudo cpupower frequency-set --governor performance
```

#### 5. Check for Thermal Throttling
```bash
# Monitor CPU temperature
sensors

# Check thermal throttling events
dmesg | grep -i thermal

# Monitor continuous temperature
watch -n 1 "sensors | grep Core"
```

### Solutions
- Use `taskset` to pin benchmarks to specific CPU cores
- Set CPU governor to `performance` mode
- Run benchmarks during low system activity
- Use multiple runs and statistical analysis
- Implement warmup periods in benchmarks

---

## High Variance in Results

### Symptoms
- Large standard deviations in benchmark results
- Coefficient of variation > 5%
- Outliers in measurement data

### Step-by-Step Diagnostics

#### 1. Analyze System Resources During Benchmark
```bash
# Start resource monitoring before benchmark
vmstat 1 > vmstat_log.txt &
iostat -x 1 > iostat_log.txt &
top -b -d 1 > top_log.txt &

# Run your benchmark
./your_benchmark

# Stop monitoring
killall vmstat iostat top
```

#### 2. Check Memory Usage Patterns
```bash
# Monitor memory usage in real-time
watch -n 1 "free -h"

# Check for memory fragmentation
cat /proc/buddyinfo

# Monitor swap usage
swapon -s
watch -n 1 "cat /proc/swaps"
```

#### 3. Monitor Cache Performance
```bash
# Check cache hit rates (requires perf)
perf stat -e cache-references,cache-misses ./your_benchmark

# Monitor cache usage
perf stat -e L1-dcache-loads,L1-dcache-load-misses ./your_benchmark
```

#### 4. Check for Context Switching
```bash
# Monitor context switches
vmstat 1 10

# Get detailed process switching info
pidstat -w 1 10

# Check scheduling statistics
cat /proc/schedstat
```

#### 5. Analyze Interrupt Activity
```bash
# Monitor interrupts
watch -n 1 "cat /proc/interrupts"

# Check interrupt distribution across CPUs
cat /proc/interrupts | head -1; cat /proc/interrupts | grep -E "(timer|eth|sda)"
```

### Solutions
- Increase number of benchmark iterations
- Use statistical methods (median, trimmed mean)
- Implement proper warmup phases
- Control system load during benchmarking
- Use CPU isolation with `isolcpus` kernel parameter

---

## Memory Leaks

### Symptoms
- Steadily increasing memory usage over time
- Out of memory errors during long-running benchmarks
- Performance degradation over time

### Step-by-Step Diagnostics

#### 1. Monitor Memory Usage Over Time
```bash
# Continuous memory monitoring
while true; do
    echo "$(date): $(free -h | grep Mem)"
    sleep 5
done > memory_usage.log

# Monitor specific process memory
pid=$(pgrep your_benchmark)
while kill -0 $pid 2>/dev/null; do
    ps -p $pid -o pid,vsz,rss,%mem,command
    sleep 1
done
```

#### 2. Use Valgrind for Memory Analysis
```bash
# Check for memory leaks
valgrind --leak-check=full --show-leak-kinds=all ./your_benchmark

# Check for memory errors
valgrind --tool=memcheck --track-origins=yes ./your_benchmark

# Monitor heap usage
valgrind --tool=massif ./your_benchmark
```

#### 3. Monitor System Memory with /proc
```bash
# Check memory info
cat /proc/meminfo

# Monitor memory maps of process
pid=$(pgrep your_benchmark)
cat /proc/$pid/maps
cat /proc/$pid/smaps

# Check process memory status
cat /proc/$pid/status | grep -E "(VmSize|VmRSS|VmData|VmStk|VmExe|VmLib)"
```

#### 4. Use System Tools for Memory Analysis
```bash
# Monitor memory usage with htop
htop -p $(pgrep your_benchmark)

# Use pmap to analyze memory layout
pmap -x $(pgrep your_benchmark)

# Monitor memory with smem
smem -p your_benchmark
```

#### 5. Check for Memory Fragmentation
```bash
# Check memory fragmentation
cat /proc/buddyinfo

# Monitor slab allocation
cat /proc/slabinfo | head -20

# Check kernel memory usage
cat /proc/meminfo | grep -E "(Slab|KernelStack|PageTables)"
```

### Solutions
- Use memory profiling tools (Valgrind, AddressSanitizer)
- Implement proper resource cleanup
- Monitor memory usage during development
- Use memory pools for frequent allocations
- Set memory limits with `ulimit -v`

---

## CPU Throttling

### Symptoms
- Decreasing performance over time
- Inconsistent CPU-bound benchmark results
- Temperature-related performance drops

### Step-by-Step Diagnostics

#### 1. Monitor CPU Frequency
```bash
# Check current CPU frequencies
watch -n 1 "cat /proc/cpuinfo | grep 'cpu MHz'"

# Monitor frequency scaling
watch -n 1 "cpupower frequency-info"

# Check frequency scaling statistics
cat /sys/devices/system/cpu/cpu0/cpufreq/stats/time_in_state
```

#### 2. Monitor Temperature
```bash
# Check CPU temperature
sensors

# Monitor temperature continuously
watch -n 1 "sensors | grep -E 'Core|Package'"

# Check thermal zone information
cat /sys/class/thermal/thermal_zone*/temp
cat /sys/class/thermal/thermal_zone*/type
```

#### 3. Check Thermal Throttling Events
```bash
# Check kernel messages for throttling
dmesg | grep -i "thermal\|throttl"

# Monitor thermal events
journalctl -f | grep -i thermal

# Check performance counters
perf stat -e thermal_throttle ./your_benchmark
```

#### 4. Monitor Power Management
```bash
# Check power management settings
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor

# Monitor turbo boost status
cat /sys/devices/system/cpu/intel_pstate/no_turbo

# Check power limits
cat /sys/class/powercap/intel-rapl/intel-rapl:0/power_limit_uw
```

### Solutions
- Set CPU governor to `performance`
- Improve system cooling
- Monitor thermal throttling during benchmarks
- Use shorter benchmark runs with cooldown periods
- Disable turbo boost for consistent results

---

## System Resource Contention

### Symptoms
- Benchmarks affected by other processes
- Inconsistent I/O performance
- Variable network performance

### Step-by-Step Diagnostics

#### 1. Monitor Overall System Activity
```bash
# Check system load and processes
htop

# Monitor I/O activity
iotop -a

# Check network activity
nethogs
```

#### 2. Identify Resource-Heavy Processes
```bash
# Find CPU-intensive processes
ps aux --sort=-%cpu | head -10

# Find memory-intensive processes
ps aux --sort=-%mem | head -10

# Monitor real-time resource usage
top -o %CPU
```

#### 3. Check I/O Contention
```bash
# Monitor disk I/O
iostat -x 1 10

# Check disk usage per process
iotop -p $(pgrep your_benchmark)

# Monitor inode usage
df -i
```

#### 4. Monitor Network Resources
```bash
# Check network connections
netstat -tuln

# Monitor network bandwidth
iftop

# Check network buffer usage
cat /proc/net/sockstat
```

#### 5. Check System Limits
```bash
# Check file descriptor limits
ulimit -n

# Check process limits
cat /proc/sys/kernel/pid_max

# Monitor system file table
cat /proc/sys/fs/file-nr
```

### Solutions
- Run benchmarks during low system activity
- Use process isolation with cgroups
- Set appropriate process priorities
- Monitor system resources during benchmarks
- Use dedicated benchmark environments

---

## Garbage Collection Issues

### Symptoms
- Periodic performance spikes in GC-enabled languages
- Memory usage patterns with saw-tooth shapes
- Inconsistent allocation-heavy benchmark results

### Step-by-Step Diagnostics

#### 1. Monitor GC Activity (Language-Specific)

##### For Go Programs:
```bash
# Enable GC trace
GODEBUG=gctrace=1 ./your_benchmark

# Monitor memory stats
GODEBUG=memstats=1 ./your_benchmark

# Profile memory usage
go tool pprof -alloc_space ./your_benchmark
```

##### For Java Programs:
```bash
# Enable GC logging
java -XX:+PrintGC -XX:+PrintGCDetails ./your_benchmark

# Monitor heap usage
jstat -gc <pid> 1s

# Get heap dump
jmap -dump:format=b,file=heap.hprof <pid>
```

#### 2. Monitor Memory Allocation Patterns
```bash
# Use perf to monitor allocations
perf record -g --call-graph dwarf ./your_benchmark
perf report

# Monitor malloc calls
strace -e malloc ./your_benchmark
```

#### 3. Check Memory Pressure
```bash
# Monitor memory pressure
cat /proc/pressure/memory

# Check OOM killer activity
dmesg | grep -i "killed process"

# Monitor swap activity
vmstat 1 10
```

### Solutions
- Tune GC parameters for your language/runtime
- Pre-allocate memory when possible
- Use object pooling for frequent allocations
- Monitor GC metrics during benchmarks
- Consider GC-friendly benchmark design

---

## General Diagnostic Commands Reference

### Quick System Overview
```bash
# System information
uname -a
cat /etc/os-release
lscpu
free -h
df -h
```

### Resource Monitoring
```bash
# CPU usage
htop
top
vmstat 1 10

# Memory usage
free -h
cat /proc/meminfo
smem

# Disk I/O
iostat -x 1 10
iotop

# Network
netstat -i
ss -tuln
```

### Performance Analysis
```bash
# System performance
perf stat ./your_benchmark
perf record -g ./your_benchmark
perf report

# Process analysis
strace -c ./your_benchmark
ltrace -c ./your_benchmark
```

### Hardware Information
```bash
# CPU information
lscpu
cat /proc/cpuinfo
cpupower frequency-info

# Memory information
dmidecode -t memory
cat /proc/meminfo

# Thermal information
sensors
cat /sys/class/thermal/thermal_zone*/temp
```

---

## Best Practices for Benchmark Stability

1. **Environment Control**
   - Use dedicated hardware for benchmarking
   - Disable unnecessary services
   - Set CPU governor to `performance`
   - Use CPU isolation with `isolcpus`

2. **Benchmark Design**
   - Implement proper warmup periods
   - Use statistical analysis of multiple runs
   - Monitor system resources during benchmarks
   - Include resource cleanup in benchmarks

3. **Monitoring**
   - Always monitor system resources during benchmarks
   - Log environmental conditions
   - Use automated analysis of benchmark variance
   - Set up alerts for unusual system behavior

4. **Reproducibility**
   - Document system configuration
   - Use version control for benchmark code
   - Record environmental conditions
   - Implement automated benchmark regression detection
