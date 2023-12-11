mkdir -p weights/GenHowTo-STATES-96h-v1

echo -n "Downloading GenHowTo config file ... "
wget https://data.ciirc.cvut.cz/public/projects/2023GenHowTo/weights/GenHowTo-STATES-96h-v1/GenHowTo_controlnet_config.json -q -O weights/GenHowTo-STATES-96h-v1/GenHowTo_controlnet_config.json
SHA256SUM=$(sha256sum weights/GenHowTo-STATES-96h-v1/GenHowTo_controlnet_config.json | cut -d' ' -f1)
if [[ ${SHA256SUM} == "de619b81fa7f43f6f0342a20cfa2031d0b9bf2d84d18ab64a902c4e1be2e6e99" ]]; then
  echo "OK ✓"
else
  echo "ERROR ✗"
  exit 1
fi

echo -n "Downloading GenHowTo backbone weights ... "
wget https://data.ciirc.cvut.cz/public/projects/2023GenHowTo/weights/GenHowTo-STATES-96h-v1/GenHowTo_sdunet.pth -q -O weights/GenHowTo-STATES-96h-v1/GenHowTo_sdunet.pth
SHA256SUM=$(sha256sum weights/GenHowTo-STATES-96h-v1/GenHowTo_sdunet.pth | cut -d' ' -f1)
if [[ ${SHA256SUM} == "9e6a5114f02322788d3e828d8c20229088b63737545fffcb26be2452baf34412" ]]; then
  echo "OK ✓"
else
  echo "ERROR ✗"
  exit 1
fi

echo -n "Downloading GenHowTo ControlNet weights ... "
wget https://data.ciirc.cvut.cz/public/projects/2023GenHowTo/weights/GenHowTo-STATES-96h-v1/GenHowTo_controlnet.pth -q -O weights/GenHowTo-STATES-96h-v1/GenHowTo_controlnet.pth
SHA256SUM=$(sha256sum weights/GenHowTo-STATES-96h-v1/GenHowTo_controlnet.pth | cut -d' ' -f1)
if [[ ${SHA256SUM} == "6354404841c14369cbc070427c0dc396571274d894e0383fb9ab544510ab97e4" ]]; then
  echo "OK ✓"
else
  echo "ERROR ✗"
  exit 1
fi
