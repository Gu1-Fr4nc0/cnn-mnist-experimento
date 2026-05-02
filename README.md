# Comparação entre CNN e MLP no MNIST

Devido a dificuldades técnicas com o ambiente R (problemas na instalação do TensorFlow e conflitos de pacotes), o experimento foi realizado em Python.
Os resultados obtidos estão de acordo com a literatura e demonstram a superioridade da CNN.

## Arquiteturas
- **CNN**: 2 camadas convolucionais (32 e 64 filtros), MaxPooling, Dropout (0.25 e 0.5), camada densa (128) e saída softmax. 30 épocas.
- **MLP**: Flatten + uma camada densa (128) + saída softmax. 30 épocas.

## Resultados
- **Acurácia CNN**: 98,5% (conforme matriz de confusão)
- **Acurácia MLP**: 97,8%

## Como reproduzir
Execute o notebook no Google Colab (runtime Python) ou rode o script `experimento_cnn_mlp.py` em um ambiente com Python 3 e TensorFlow instalado.
