# ============================================================
# Experimento CNN vs MLP – MNIST com matrizes de confusão
# Código final e corrigido (sem y_train_raw)
# ============================================================

# 1. Pacotes
if (!require("keras3")) install.packages("keras3")
if (!require("caret")) install.packages("caret")
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(reshape2)) install.packages("reshape2")
library(keras3)
library(caret)
library(ggplot2)
library(reshape2)

# 2. Dados MNIST
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test  <- mnist$test$x
y_test  <- mnist$test$y

# Normalização e reshape
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1)) / 255
x_test  <- array_reshape(x_test,  c(nrow(x_test), 28, 28, 1)) / 255

# One‑hot para treino
y_train_cat <- to_categorical(y_train, 10)
y_test_cat  <- to_categorical(y_test, 10)

# Verificação rápida
cat("x_train valores entre", min(x_train), "e", max(x_train), "\n")
cat("Exemplo de rótulo original (antes do one-hot):", y_train[1:5], "\n")

# 3. CNN
model_cnn <- keras_model_sequential() %>%
  layer_conv_2d(32, c(3,3), activation = 'relu', input_shape = c(28,28,1)) %>%
  layer_conv_2d(64, c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(c(2,2)) %>%
  layer_dropout(0.25) %>%
  layer_flatten() %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dropout(0.5) %>%
  layer_dense(10, activation = 'softmax')

model_cnn %>% compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

cat("\n=== Treinando CNN ===\n")
history_cnn <- model_cnn %>% fit(x_train, y_train_cat,
                                 batch_size = 128, epochs = 12,
                                 validation_split = 0.2, verbose = 1)

scores_cnn <- model_cnn %>% evaluate(x_test, y_test_cat, verbose = 0)
cat(sprintf("Acurácia CNN: %.4f\n", scores_cnn[[2]]))

# 4. MLP
model_mlp <- keras_model_sequential() %>%
  layer_flatten(input_shape = c(28,28,1)) %>%
  layer_dense(128, activation = 'relu') %>%
  layer_dense(10, activation = 'softmax')

model_mlp %>% compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = 'accuracy')

cat("\n=== Treinando MLP ===\n")
history_mlp <- model_mlp %>% fit(x_train, y_train_cat,
                                 batch_size = 128, epochs = 12,
                                 validation_split = 0.2, verbose = 1)

scores_mlp <- model_mlp %>% evaluate(x_test, y_test_cat, verbose = 0)
cat(sprintf("Acurácia MLP: %.4f\n", scores_mlp[[2]]))

# 5. Matrizes de confusão
plot_cm <- function(cm, title) {
  cm_df <- as.data.frame(as.table(cm))
  colnames(cm_df) <- c("Predito", "Real", "Freq")
  ggplot(cm_df, aes(x = Predito, y = Real, fill = Freq)) +
    geom_tile() + geom_text(aes(label = Freq), size = 4) +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(title = title, x = "Classe Predita", y = "Classe Real") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# CNN
pred_cnn <- model_cnn %>% predict(x_test)
pred_cnn_classe <- apply(pred_cnn, 1, which.max) - 1
cm_cnn <- confusionMatrix(as.factor(pred_cnn_classe), as.factor(y_test))
cat("\nMatriz de confusão - CNN (valores):\n")
print(cm_cnn$table)
p_cnn <- plot_cm(cm_cnn$table, "Matriz de Confusão - CNN")
print(p_cnn)
ggsave("confusion_matrix_cnn.png", p_cnn, width = 8, height = 6, dpi = 300)

# MLP
pred_mlp <- model_mlp %>% predict(x_test)
pred_mlp_classe <- apply(pred_mlp, 1, which.max) - 1
cm_mlp <- confusionMatrix(as.factor(pred_mlp_classe), as.factor(y_test))
cat("\nMatriz de confusão - MLP (valores):\n")
print(cm_mlp$table)
p_mlp <- plot_cm(cm_mlp$table, "Matriz de Confusão - MLP")
print(p_mlp)
ggsave("confusion_matrix_mlp.png", p_mlp, width = 8, height = 6, dpi = 300)

cat("\n✅ Concluído. Matrizes salvas como:\n   confusion_matrix_cnn.png\n   confusion_matrix_mlp.png\n")