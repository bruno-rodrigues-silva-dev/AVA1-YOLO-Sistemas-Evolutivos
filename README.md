# AVA1 - YOLO Sistemas Evolutivos

Projeto desenvolvido para a disciplina de Sistemas Evolutivos e Aplicados à Robótica.

## Objetivo

Utilizar o modelo YOLO para realizar detecção de objetos em imagem utilizando Python e visão computacional.

## Tecnologias utilizadas

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- Google Colab

## Código

```python
import cv2
import numpy as np
import time
import requests
from ultralytics import YOLO
from google.colab.patches import cv2_imshow

url = "LINK_DA_IMAGEM"

response = requests.get(url)

image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

model = YOLO("yolo26l.pt")

start = time.perf_counter()

results = model(img)

end = time.perf_counter()

tempo = end - start

print(f"Tempo de inferência: {tempo:.4f} segundos")

annotated = results[0].plot()

cv2_imshow(annotated)
```

## Resultado

O sistema realiza a detecção de objetos na imagem e exibe o resultado anotado com o modelo YOLO.

## Autor

Bruno Rodrigues da Silva
