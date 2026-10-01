#!/usr/bin/env bash
set +m

nodes=(
    bufflehead brent barnacle aylesbury cackling crested 
    barbury canada gadwall eider gressingham goosander 
    mallard mandarin harlequin pintail scaup ruddy 
    scoter pochard shoveler shelduck smew wigeon
)

# 1. Static header
printf "%-15s %-12s %-50s %-s\n" "NODE" "OS/CPU LOAD" "USERS / SCRIPT" "GPU STATUS / DETAILS"
echo "---------------------------------------------------------------------------------------------------------------------------------"

MAX_CONCURRENT_JOBS=12

# 2. Stream block for parallel outputs, sorted at the end
{
    for node in "${nodes[@]}"; do
        while [ $(jobs -rp | wc -l) -ge $MAX_CONCURRENT_JOBS ]; do
            sleep 0.05
        done

        (
            node=$(echo "$node" | tr -d '\r')
            domain=".cs.ucl.ac.uk"
            linux_host="${node}-l${domain}"
            windows_host="${node}-w${domain}"

            # --- PHASE 1: LINUX EXPERIMENTS ---
            if ping -c 1 -W 1 "$linux_host" &>/dev/null; then
                err_file="/tmp/ssh_err_${node}_$$"

                res=$(ssh -T -o RemoteCommand=none -o ConnectTimeout=2 -o StrictHostKeyChecking=no "$linux_host" 'bash -s' 2>"$err_file" << 'REMOTE_EOF'
                    load=$(uptime | awk -F"load average:" '{print $2}' | awk -F, '{print $1}' | xargs)

                    # Snapshot BEFORE any of our own filtering helpers exist.
                    raw=$(ps -eo uid=,user=,args= 2>/dev/null)

                    procs=$(printf '%s\n' "$raw" | awk '
                        BEGIN {
                            # Interpreters we treat as a "script" if no file ext is found
                            split("python python3 python2 ipython jupyter Rscript julia node deno bun ruby perl matlab", interp, " ")
                            for (i in interp) interp_set[interp[i]] = 1
                            # Shells we do NOT report as a job (they just mean "logged in")
                            split("bash sh zsh csh tcsh ksh fish dash login", shells, " ")
                            for (i in shells) shell_set[shells[i]] = 1
                        }
                        {
                            uid = $1
                            user = $2
                            sub(/^[^ ]+ +[^ ]+ +/, "", $0)   # strip uid and user, keep args
                            args = $0

                            # Only real, interactive users
                            if (uid < 1000 || uid == 65534) next
                            if (args == "" || args ~ /^\(sd-pam\)/) next
                            # Skip SSH/session plumbing that is still owned by a real user
                            if (args ~ /^sshd: / || args ~ /^ssh-agent/ || args ~ /^dbus-daemon/) next

                            n = split(args, toks, / +/)
                            script = ""

                            # Priority 1: a token ending in a known script extension
                            for (i = 1; i <= n; i++) {
                                t = toks[i]
                                if (t ~ /\.(py|sh|pl|rb|R|jl|ipynb)$/) {
                                    sub(/.*\//, "", t)
                                    script = t
                                    break
                                }
                            }

                            # Priority 2: first token is a recognized interpreter / tool
                            if (script == "") {
                                t = toks[1]
                                sub(/.*\//, "", t)
                                sub(/^-/, "", t)
                                if (t in interp_set) script = t
                            }

                            # Priority 3: fall back to first token basename, but skip bare shells
                            if (script == "") {
                                t = toks[1]
                                sub(/.*\//, "", t)
                                sub(/^-/, "", t)
                                if (t == "" || t in shell_set) next
                                script = t
                            }

                            print user ":" script
                        }
                    ' | sort -u | tr '\n' ' ' | sed 's/ $//')

                    if [ -z "$procs" ]; then
                        my_jobs="None"
                    else
                        my_jobs="$procs"
                    fi

                    # GPU info — guard against nvidia-smi printing a text error on stdout
                    if command -v nvidia-smi &>/dev/null; then
                        gpu_raw=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
                                    --format=csv,noheader,nounits 2>/dev/null)
                        if [ -z "$gpu_raw" ]; then
                            gpu_info="No GPU Hardware"
                        else
                            gpu_info=$(printf '%s\n' "$gpu_raw" \
                                | awk -F', ' '/^[0-9]+, [0-9]+, [0-9]+$/ {printf "[%s%% Util, %s/%s MiB] ", $1, $2, $3}')
                            [ -z "$gpu_info" ] && gpu_info="No GPU Hardware"
                        fi
                    else
                        gpu_info="No GPU Hardware"
                    fi

                    printf "%s\t%s\t%s\n" "$load" "$my_jobs" "$gpu_info"
REMOTE_EOF
                )

                ssh_err=$(cat "$err_file" 2>/dev/null | tr '\r\n' '  ' | xargs)

                if [ -z "$res" ]; then
                    [ -z "$ssh_err" ] && ssh_err="SSH timed out or failed"
                    out_line=$(printf "\033[31m%-15s %-12s %-50s %-s\033[0m\n" "$node" "LINUX (ERR)" "-" "$ssh_err")
                    echo "1_999_999.99|$out_line"
                else
                    IFS=$'\t' read -r load_val my_job_val gpu_val <<< "$res"

                    [[ ! "$load_val" =~ ^[0-9.]+$ ]] && load_val="0.00"
                    [ -z "$my_job_val" ] && my_job_val="None"
                    [ -z "$gpu_val" ] && gpu_val="No GPU Hardware"

                    if [[ "$gpu_val" =~ ([0-9]+)% ]]; then
                        gpu_util="${BASH_REMATCH[1]}"
                    else
                        gpu_util=0
                    fi

                    sort_key=$(printf "1_%03d_%06.2f" "$gpu_util" "$load_val")

                    if [[ "$gpu_val" == *"0% Util, 0/"* || "$gpu_val" == "No GPU Hardware" ]] \
                       && awk -v l="$load_val" 'BEGIN{exit !(l < 0.5)}'; then
                        out_line=$(printf "\033[32m%-15s %-12s %-50s %-s\033[0m\n" "$node" "$load_val" "$my_job_val" "$gpu_val")
                    elif [ "$my_job_val" != "None" ]; then
                        out_line=$(printf "%-15s %-12s \033[36m%-50s\033[0m %-s\n" "$node" "$load_val" "$my_job_val" "$gpu_val")
                    else
                        out_line=$(printf "%-15s %-12s %-50s %-s\n" "$node" "$load_val" "$my_job_val" "$gpu_val")
                    fi
                    echo "$sort_key|$out_line"
                fi
                rm -f "$err_file"

            # --- PHASE 2: WINDOWS HOUSES ---
            elif ping -c 1 -W 1 "$windows_host" &>/dev/null; then
                out_line=$(printf "\033[33m%-15s %-12s %-50s %-s\033[0m\n" "$node" "WINDOWS" "-" "Running Windows (-w suffix active)")
                echo "2_000_000.00|$out_line"

            # --- PHASE 3: OFFLINE BOXES ---
            else
                out_line=$(printf "\033[31m%-15s %-12s %-50s %-s\033[0m\n" "$node" "DOWN" "-" "Offline / Unreachable")
                echo "3_000_000.00|$out_line"
            fi
        ) &
    done
    wait
} | sort | cut -d'|' -f2-