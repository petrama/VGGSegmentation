# VGGSegmentation
Segmentation vgg16 fcn - cityscapes
Priprema skupa
--------------
skripta prepare_dataset_downsampled.py

Iz slika cityscapesa izrezuje haubu automobila, i smanjuje sliku na željenu rezoluciju, to zapisuje u tfrecords formatu.
Treba zadati putanju do cityscapesa, izlazni direktorij gdje će se spremati tfrecordsi i zadati željenu rezoluciju.

Priprema težina vgg-a
-----------------------
Da bi se model mogao fine-tuneati treba na disku imati spremljene težine mreže (prethodno naučene na nekom drugom skupu).
One se mogu skinuti s interneta u raznim formatima. 

Ja sam ih imala spremljene u sljedećim datotekama:
conv1_1_biases.bin
conv1_1_weights.bin
conv1_2_biases.bin
conv1_2_weights.bin
conv2_1_biases.bin
conv2_1_weights.bin
conv2_2_biases.bin
conv2_2_weights.bin
conv3_1_biases.bin
conv3_1_weights.bin
conv3_2_biases.bin
conv3_2_weights.bin
conv3_3_biases.bin
conv3_3_weights.bin
conv4_1_biases.bin
conv4_1_weights.bin
conv4_2_biases.bin
conv4_2_weights.bin
conv4_3_biases.bin
conv4_3_weights.bin
conv5_1_biases.bin
conv5_1_weights.bin
conv5_2_biases.bin
conv5_2_weights.bin
conv5_3_biases.bin
conv5_3_weights.bin
fc6_biases.bin
fc6_weights.bin
fc7_biases.bin
fc7_weights.bin
fc8_biases.bin
fc8_weights.bin

Ako će se težine učitavati iz ckpt. datoteke npr vgg_16.ckpt, onda će
i u kodu trebati mjenjati metodu create_init_op unutar model.py

Konfiguracija
--------------

config/cityscapes.py - primjer fajla s konfiguracijom za treniranje

Treba promjeniti putanje

model_path da pokazuje do py fajla s definicijom modela (primjer za takve dvije defincije su model.py i model2.py)


dataset_dir - da pokazuje do foldera s prethodno pripremljenim tfrecordsima (koji sadrzi subdirektorije train i val)

treba paziti pri razlicitim rezolucijama da se promjene zastavice img_width i height

ostale zastavice se većinom odnose na treniranje modela to mjenjati prema potrebi.

subsample_factor zastavica bi označavala faktor za koji se rezolucija mape smanji na kraju mreže.
Taj faktor će ovisiti o samome modelu koji se trenira, ako model ima tri pooling sloja 2*2 svaki taj sloj će sliku smanjiti za dva puta pa će ukupno smanjnjenje biti za faktor osam



train.py - skripta koja pokreće skriptu treniranja, nakon svake epohe model se evaluira na skupu za validaciju.



