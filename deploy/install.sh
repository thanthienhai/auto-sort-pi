#!/usr/bin/env bash
# Cài đặt auto-sort-pi làm systemd service trên Raspberry Pi 4 (§17 task.md).
#
# Giả định: mã nguồn đã được sao chép/clone vào /opt/auto-sort-pi (chỉnh
# INSTALL_DIR bên dưới nếu khác), và script chạy bằng sudo trên chính con Pi.
#
# KHÔNG tự động chạy model validation / calibration — đây là bước thủ công
# bắt buộc trước khi enable service thật sự (xem README.md § Production
# checklist). Script này chỉ lo phần "cài đặt hạ tầng".
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/auto-sort-pi}"
SERVICE_NAME="auto-sort-pi"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$EUID" -ne 0 ]]; then
  echo "Cần chạy bằng sudo (để cài systemd unit + tạo group truy cập serial/camera)." >&2
  exit 1
fi

echo "==> Đồng bộ mã nguồn vào ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
rsync -a --exclude='.venv' --exclude='.git' --exclude='__pycache__' --exclude='logs/' \
  "${REPO_DIR}/" "${INSTALL_DIR}/"

echo "==> Tạo virtualenv + cài dependency"
python3 -m venv "${INSTALL_DIR}/.venv"
"${INSTALL_DIR}/.venv/bin/pip" install --upgrade pip
"${INSTALL_DIR}/.venv/bin/pip" install -r "${INSTALL_DIR}/requirements.txt"

echo "==> Thêm user pi vào group dialout/video (nếu chưa có)"
usermod -aG dialout,video pi || true

if [[ ! -f "${INSTALL_DIR}/deploy/auto-sort-pi.env" ]]; then
  cp "${INSTALL_DIR}/deploy/auto-sort-pi.env.example" "${INSTALL_DIR}/deploy/auto-sort-pi.env"
fi

echo "==> Cài systemd unit"
cp "${INSTALL_DIR}/deploy/auto-sort-pi.service" "/etc/systemd/system/${SERVICE_NAME}.service"
systemctl daemon-reload

cat <<EOF

Đã cài xong hạ tầng. TRƯỚC KHI enable service, bắt buộc phải:

  1. Đặt model .tflite thật vào ${INSTALL_DIR}/models/ và validate:
       ${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/tools/validate_model.py --model <path>
  2. Calibration thật (không phải identity matrix):
       ${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/tools/calibrate.py
       ${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/tools/test_calibration.py --points-file <check_points.json>
  3. Kiểm tra health trước khi enable:
       ${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/tools/health.py

Sau khi đạt cả 3 bước trên:
  sudo systemctl enable --now ${SERVICE_NAME}
  sudo journalctl -u ${SERVICE_NAME} -f
EOF
