# prepare data

```powershell
# ===============prepare DSText============
cd Data
mkdir DSText
cd DSText
mkdir train test
sh ../tools/get_data/get_DSText.sh   # v1和测试集
# TODO: 从谷歌云盘下载V2
unzip V2_Ann_Train.zip -d train
unzip V2_Video_Train.zip -d train

# ==============prepare COCOTextv2=========

```