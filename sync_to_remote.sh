#!/usr/bin/env bash
# sync_to_remote.sh —— 将本项目未被 VCS 忽略的文件 rsync 同步到远端机器。详见 -h。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

host="fastai63"
port=22
# dest 默认为项目绝对路径去掉 $HOME 前缀后的相对路径，落在远端用户家目录下；
# 若项目不在 $HOME 下则回退为绝对路径。
dest="$SCRIPT_DIR"
if [[ -n "${HOME:-}" && "$SCRIPT_DIR" == "$HOME"/* ]]; then
    dest="${SCRIPT_DIR#$HOME/}"
fi
includes=()
excludes=()

usage() {
    cat <<'EOF'
用法: sync_to_remote.sh [选项]

将本项目未被 VCS 忽略的文件通过 rsync 同步到远端机器。
文件清单来源：`rg --files`（默认遵守 .gitignore），因此无需手动解析忽略规则。
额外需要同步（即便被 .gitignore 忽略）的条目用 --include 指定；
需要额外排除的条目用 --exclude 指定。

选项:
  --host HOST      远端主机，默认 fast63
  --port PORT      SSH 端口，默认 22
  --dest PATH      远端目标路径，默认为本地项目路径去掉 $HOME 前缀的相对路径
  --include PATH   额外同步的文件/目录（可重复，即便被忽略也强制加入）
  --exclude PAT    额外排除的文件/目录（可重复，rsync glob 模式）
  -h, --help       显示本帮助
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)   host="${2:?--host 需要一个参数}"; shift 2 ;;
        --port)   port="${2:?--port 需要一个参数}"; shift 2 ;;
        --dest)   dest="${2:?--dest 需要一个参数}"; shift 2 ;;
        --include) includes+=("${2:?--include 需要一个参数}"); shift 2 ;;
        --exclude) excludes+=("${2:?--exclude 需要一个参数}"); shift 2 ;;
        -h|--help) usage ;;
        *) echo "未知选项: $1（使用 -h 查看帮助）" >&2; exit 1 ;;
    esac
done

command -v rsync >/dev/null || { echo "未找到 rsync，请先安装" >&2; exit 1; }
command -v rg    >/dev/null || { echo "未找到 rg (ripgrep)，请先安装" >&2; exit 1; }

# 1) 构建文件清单：rg --files 列出未被 VCS 忽略的文件；再追加 --include 条目
tmpfile="$(mktemp)"
trap 'rm -f "$tmpfile"' EXIT
rg --files > "$tmpfile"
if ((${#includes[@]})); then
    printf '%s\n' "${includes[@]}" >> "$tmpfile"
fi

# 2) 组装 rsync 参数
rsync_opts=(-avz --files-from="$tmpfile")
if ((${#excludes[@]})); then
    for pat in "${excludes[@]}"; do
        rsync_opts+=(--exclude="$pat")
    done
fi
rsync_opts+=(-e "ssh -p ${port}")

# 3) 同步（源为当前项目根目录，目标为 host:dest）
echo "同步到 ${host}:${dest} (端口 ${port})，共 $(wc -l < "$tmpfile") 个条目"
rsync "${rsync_opts[@]}" ./ "${host}:${dest}"

