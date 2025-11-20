# Self-Supervised Learning on CIFAR-10  
### Contrastive Learning • Barlow Twins • Transfer Learning

Este proyecto implementa un pipeline completo de **aprendizaje supervisado** y **auto-supervisado (SSL)** usando el dataset **CIFAR-10**, explorando cómo cambia el rendimiento según la cantidad de datos etiquetados y cómo un modelo auto-supervisado puede mejorar la generalización.

---

## 🎯 Objetivos del Proyecto

- Entrenar una red neuronal **supervisada** con distintos porcentajes de datos etiquetados.
- Implementar un pipeline de **Self-Supervised Learning (SSL)** estilo **Barlow Twins**.
- Usar augmentaciones avanzadas para generar dos vistas de la misma imagen.
- Comparar:
  - Entrenamiento desde cero
  - Entrenamiento con pesos preentrenados (ResNet18 pretrained)
  - Fine-tuning de un backbone auto-supervisado
- Visualizar pérdidas, accuracy y el efecto del SSL.

---

## 🧠 ¿Qué se aprende aquí?

### ✔️ 1. Cargar y manipular CIFAR-10  
Uso de `torchvision.datasets.CIFAR10`, `Dataset`, `DataLoader` y normalización.

### ✔️ 2. Entrenar un modelo supervisado  
- Arquitectura basada en **ResNet18**
- Optimización con Adam
- Cálculo de cross entropy y accuracy

### ✔️ 3. Experimentos con distintas cantidades de datos  
Se entrenan modelos con:
pctgs = [0.01, 0.1, 1.0]

Comparando performance según la disponibilidad de etiquetas.

### ✔️ 4. Augmentaciones con Albumentations  
Incluyendo:

- RandomResizedCrop  
- HorizontalFlip  
- ColorJitter  
- ToGray  
- Solarize  

### ✔️ 5. Self-Supervised Learning (tipo Barlow Twins)  
Implementación del loss contrastivo usando:

- Dos vistas randomizadas
- Normalización batch-wise
- Cross-correlation matrix
- Penalización diagonal vs off-diagonal

### ✔️ 6. Fine-Tuning desde un backbone Self-Supervised  
El backbone SSL se guarda en TorchScript:

```python
torch.jit.script(SSLmodel.backbone).save("SSLbackbone.pt")

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
