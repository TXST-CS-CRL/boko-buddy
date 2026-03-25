## Program Execution and Performance Evaluation in Linux
CS3339: Computer Architecture

Apan Qasem [\<apan@txstate.edu\>](apan@txstate.edu)

### Description 

A simple introduction to command-line program execution and performance evaluation. Covers the following

   * basic Linux commands 
   * command-line compilation with `gcc`
   * performance profiling with `perf`. 

This tutorial is designed for students taking CS3339. The listed remote servers are available to
Texas State students registered in any CS course. The git repository is accessible to students in
CS3339. 

### Outline 

* [Environment Set-up](#env)
* [Obtaining code samples](#git)
* [Building and executing](#gcc)
* [Measuring performance](#time)
* [Exploring a new architecture: POWER8](#capi) 
* [Getting performance insight via `perf`](#perf)
* [Additional resources](#addtl)


### <a name="env"></a>Environment Set-up
   
**(i) Login to remote Linux system** 

To connect to the remote servers from a Windows machine, use [PuTTY](https://www.putty.org/) or
similar app. To connect from a Linux or a Mac use the terminal. From a terminal window, login to one
of the CS servers using `ssh`.  

```
ssh netid@zeus.cs.txstate.edu
```

`zeus` is one of the two _public_ CS servers available for remote access to all CS students. The
other one is `eros`. Both can be accessed from outside the university firewall. 

The above command will prompt you to verify your credentials using one of four MFA methods. If you have not set up MFA with TXST, please do so using the [DUO Enrollment Guideline](https://services.txst.edu/TDClient/39/ITAC/Requests/ServiceDet?ID=371). If the userid on your local Mac or Linux system is the same as your netid, then you can simply type `ssh
server_name.cs.txstate.edu`.  

The department and the [CRL](compilers.cs.txstate.edu) research group maintains many other servers
for which you can request access. For this class, you will all be granted access to
`capi.cs.txstate.edu` an IBM POWER8 server. 

**(ii) Getting around a Linux system: files and directories** 

List files in current directory 

```
ls
```

Almost all Linux commands will accept a set of options that can be used to control its
behavior. Options are passed at the command-line using a `-` followed by a single character. The
character corresponds to the particular option that you want enforced. Here are few useful flags for
the `ls` command.
 
List files in current directory including hidden files 
```
ls -a
```

List files and attributes in current directory reverse chronological order 

```
ls -ltr 
```

Set up a working directory for the experiments. Create a directory
```
mkdir cs3339
```

Change to working directory
```
cd cs3339
```

Change to home directory
```
cd
```

Change to working directory
```
cd cs3339
```

**(iii) Getting familiar with the environment**

Get information about the OS and architecture. 

```
uname 
```

Get more information (-a stands for all)
``` 
uname -a
```
See who else is logged on and what they are doing 
``` 
w
```

Look up specific user 
```
finger 
```

This command has been disabled on `zeus` and `eros` by the system administrator 


List the programs (i.e., processes) that are currently running and report resource usage 
 
```
top
```

Get CPU information 

```
cat /proc/cpuinfo
```

Get memory information 

```
cat /proc/meminfo
```

### <a name="git"></a>Obtaining code samples

Clone the course git repo on this server. 

```
git clone https://github.com/apanqasem/CS3339.git ~/CS3339.git
```

Copy the matrix-vector multiplication code (matvec.c) to your working directory 

```
cp ~/CS3339.git/code_samples/matvec.c .
```

### <a name="gcc"></a>Building and executing 

Compile a C++ source. The `-c` option tells GCC to only _compile_ the source. This option
doesn't perform any _linking_ and therefore doesn't result in a program that we can execute. 
```
gcc -c matvec.c
ls -ltr 
```

If we want to build the executable then we need to omit the `-c` flag. 
```
 gcc -o matvec matvec.c
 ls -ltr 
```

There are many things happening behind scenes when going from a source to a binary executable. We
can inspect these steps by using the `-v` option. 
```
gcc -v -o matvec matvec.c
```

The compiler's main job is to just generate the human-readable assembly. Subsequent steps are
conducted by tools (e.g., assembler, linker) that are packaged with the compiler. If we want to just
generate the assembly code then we can pass the `-S` option. 

```
gcc -S matvec.c
ls -ltr 
```

Take a look at the relative size of the source, asssembly, object and executable files. 

**Do these sizes seem reasonable to you?**

To execute a program from the command-line, we simply need the name of the executable and its
location (_path_, in Linux terminology). 

```
./matvec 10000 5
```

The matvec program takes two command-line arguments which correspond to the size of the data set and
number of times the computation is to be run. 
    
### <a name="time"></a>Measuring performance 

**How do we measure the performance of a program?** 

We can use the `time` command to get a rough measure of the execution time. The terms _execution
time_ and _running time_ are synonymous. _Runtime_ means something different! 

```
time ./matvec 10000 5
```

The `time` command reports three numbers. `real` time is the time that has elapsed during the
execution of the program. `user` time is the actual time the program is running on the
processor. `sys` is the time when the _system_ is doing some work either on behalf of this program
or some program. Often `real` time is roughly equal to `user` time + `sys` time 

**Is the matvec code performing well?**

The answer is, we don't know! Just like at a single number doesn't tell us much. We need some basis
for comparison (more on this in the lecture).


Compilers implement many optimizations to try to improve the performance of a program. We can
specify the level of optimizations using the `-O` flags. By default, `gcc` will apply only a few
optimizations. This is referred to as level `-O0`. Let's rebuild the program at the highest optimization
level `-O3`. 

```
gcc -o matvec -O3 matvec.c
time ./matvec 10000 5
```

Now, we can say that the version of `matvec` without any optimizations (_-O0_) definitely has
worse performance than the optimized one. 

We can check the optimizations that are being applied at `-O3` with the `--help=optimizers` option. 
```
gcc -Q -O3 --help=optimizers
```

### <a name="capi"></a>Exploring a new architecture: POWER8 

Log in to `capi.cs.txstate.edu` using ssh. capi is not visible outside the firewall. Accounts will
be created for all of you. 
    
``` 
ssh capi.cs.txstate.edu
```

Check out the OS and architecture information on capi. 

```
uname -a
cat /proc/cpuinfo
cat /proc/meminfo
```

Generate assembly code for matvec

```
gcc -S matvec.c
```

Notice how the instructions are different from those on zeus which is an x86 machine. 

Build the executable with full optimization and measure its performance. 

```
gcc -o matvec -O3 matvec.c
time ./matvec 10000 5
```

### <a name="perf"></a>Getting performance insight via `perf`

Most recent Linux distributions come with a performance evaluation tool called `perf`. This tool
probes the underlying hardware performance counters and can measure a variety of events
during program execution. More on this later in the semester. 

To use the `perf` tool, simply type `perf` followed by a `perf` command. The `list` command lists
the available performance events on this architecture. 

```
perf list
```

The above only lists the _named_ events. Typically there are hundreds more on the system. 

To get a basic profile we can use the `stat` command, followed by the name the of the program we
want to profile. The program can be one that you wrote or any Linux command. 

```
perf stat ls
```

Now let's profile the matvec program. Note, for `matvec` we need to indicate the path of the
executable and the command-line arguments the `stat` command. 

```
perf stat ./matvec 10000 5
```

`perf` reports not only the execution time, at a better resolution, but also tells us what's
happening inside the processor. This gives us insight about potential performance bottlenecks. Most
of the items listed in the `perf` output (e.g., pipeline stalls, mis-predicted branches) are things
we will talk about later in the semester. 

### <a name="addtl"></a>Additional resources

Bourne Again SHell (BASH) 

Bash scripting cheatsheet: https://devhints.io/bash  
Bash guide: http://mywiki.wooledge.org/BashGuide  
"The ultimate guide to Linux for Windows users": https://www.dedoimedo.com/computers/ultimate-linux-guide-for-windows-users.html  
A random assortment of shell scripts: https://github.com/chaseleif/shellscripts  
