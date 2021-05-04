# No arguments, just use bash:
if [ -z "$1" ]; then
  echo "No option provided"
  bash
  exit
fi
echo "Option provided: $1"
