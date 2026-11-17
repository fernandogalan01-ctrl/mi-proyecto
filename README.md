# Asistente Conversacional Multimodal con Conciencia Espacial

Este repositorio contiene el código y la documentación de mi proyecto para el grado Sistemas Inteligentes (UIE). 

El objetivo principal de este sistema es ofrecer un asistente de navegación en interiores 100% offline y enfocado en la privacidad, diseñado especialmente para entornos donde las soluciones en la nube no son una opción viable (ya sea por latencia o por proteger la privacidad de espacios domésticos como baños o habitaciones).

---

## Instalación y puesta en marcha

Para ejecutar el proyecto en local (probado en Python 3.10+):

1. **Clonar el repositorio y entrar en la carpeta:**
   ```bash
   git clone [https://github.com/fernandogalan01-ctrl/mi-proyecto.git](https://github.com/fernandogalan01-ctrl/mi-proyecto.git)
   cd mi-proyecto
   ```
## ¿Cómo está organizado el proyecto?

Para mantener el código ordenado y separar responsabilidades, la estructura de carpetas es la siguiente:

```text
mi-proyecto/
├── model/                  # Pesos de los modelos
│   ├── best.pt             # Mi modelo custom (entrenado para barreras arquitectónicas)
│   └── yolov8n.pt          # Modelo global (COCO / mobiliario estándar)
├── my-dataset/             # Datos y configuración para el modelo de visión
│   ├── train/ / val/ / test/
│   └── custom_ds.yaml
├── src/                    # Código fuente principal
│   ├── nlp/                # Reconocimiento y síntesis de voz (Whisper y TTS)
│   ├── reasoning/          # Grafo de conocimiento y lógica espacial
│   ├── vision/             # Detección YOLOv8 y supresión de conflictos (NMS)
│   ├── main.py             # Script principal que une todo el bucle
│   └── train_model.py      # Script de entrenamiento / ajuste
└── requirements.txt        # Dependencias de Python
