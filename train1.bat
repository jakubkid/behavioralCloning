python trainTheModel.py --dataPath ./Record,./Recovery,./RecordOpposite,./Record1,./Recovery1,./Record1Opposite --epochs 25 --offset 0.15 --output LaNet1All25ep0_15 %*
python trainNvidia.py --dataPath ./Record,./Recovery,./RecordOpposite,./Record1,./Recovery1,./Record1Opposite --epochs 25 --offset 0.15 --output nvidAll25ep0_15 %*
python trainNvidiaDrop.py --dataPath ./Record,./Recovery,./RecordOpposite,./Record1,./Recovery1,./Record1Opposite --epochs 25 --offset 0.15 --output nvidAllDrp25ep0_15 %*
