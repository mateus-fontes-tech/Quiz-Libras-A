# Importa a biblioteca OpenCV para trabalhar com imagens
import cv2

# Importa o NumPy para trabalhar com arrays e cálculos
import numpy as np

# Importa o módulo OS para trabalhar com arquivos e pastas
import os

# Importa o TensorFlow para criar e treinar a rede neural
import tensorflow as tf

# Importa a função responsável por dividir os dados
from sklearn.model_selection import train_test_split

# Importa as métricas usadas para avaliar o modelo
from sklearn.metrics import classification_report, confusion_matrix


# =====================================================
# CONFIGURAÇÃO DA GPU
# =====================================================

# Procura GPUs disponíveis no computador
gpus = tf.config.list_physical_devices('GPU')

# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título da verificação
print("VERIFICAÇÃO DA GPU")

# Imprime outro separador
print("========================================")

# Verifica se alguma GPU foi encontrada
if gpus:

    # Informa que uma ou mais GPUs foram encontradas
    print("GPU(s) detectada(s):")

    # Percorre todas as GPUs encontradas
    for gpu in gpus:

        # Mostra as informações da GPU atual
        print(gpu)

    # Tenta configurar a GPU
    try:

        # Percorre novamente todas as GPUs
        for gpu in gpus:

            # Configura o crescimento gradual da memória da GPU
            tf.config.experimental.set_memory_growth(

                # GPU que está sendo configurada
                gpu,

                # Ativa o crescimento gradual da memória
                True
            )

        # Informa que o TensorFlow reconheceu a GPU
        print("\nTensorFlow detectou a GPU.")

        # Informa que a GPU poderá ser usada no treinamento
        print("O treinamento poderá utilizar a GPU.")

    # Captura um possível erro de configuração
    except RuntimeError as e:

        # Informa que ocorreu um erro
        print("\nErro ao configurar a GPU:")

        # Mostra o erro encontrado
        print(e)

# Executado quando nenhuma GPU é encontrada
else:

    # Informa que nenhuma GPU foi detectada
    print("NENHUMA GPU FOI DETECTADA.")

    # Informa que a CPU será utilizada
    print("O treinamento será executado pela CPU.")


# =====================================================
# CONFIGURAÇÕES
# =====================================================

# Define o diretório onde estão as imagens do dataset
DIRETORIO = r'C:\programacao\Python\redes_neorais\treinamento_IA\imagens'

# Define que as imagens terão 128x128 pixels
IMG_SIZE = 128

# Define uma semente para manter resultados reproduzíveis
SEED = 42

# Define a semente aleatória utilizada pelo NumPy
np.random.seed(SEED)

# Define a semente aleatória utilizada pelo TensorFlow
tf.random.set_seed(SEED)


# =====================================================
# CARREGAMENTO DAS IMAGENS
# =====================================================

# Cria uma lista vazia para armazenar as imagens
imagens = []

# Cria uma lista vazia para armazenar os rótulos
labels = []

# Cria uma lista ordenada com as classes do dataset
classes = sorted([

    # Representa cada pasta encontrada
    pasta

    # Percorre os itens existentes no diretório
    for pasta in os.listdir(DIRETORIO)

    # Verifica se o item encontrado é uma pasta
    if os.path.isdir(

        # Monta o caminho completo da pasta
        os.path.join(DIRETORIO, pasta)
    )
])

# Pula uma linha e imprime um separador
print("\n========================================")

#  
print("CLASSES ENCONTRADAS")

# Imprime outro separador
print("========================================")

# Exibe todas as classes encontradas
print(classes)

# Cria um dicionário associando cada classe a um número
classe_para_label = {

    # Usa o nome da classe como chave e o índice como valor
    classe: indice

    # Percorre as classes juntamente com seus índices
    for indice, classe in enumerate(classes)
}


# =====================================================
# LEITURA DAS IMAGENS
# =====================================================

# Percorre todas as classes encontradas
for classe in classes:

    # Cria o caminho da pasta correspondente à classe
    pasta_classe = os.path.join(

        # Diretório principal das imagens
        DIRETORIO,

        # Nome da classe atual
        classe
    )

    # Percorre os arquivos existentes na pasta da classe
    for arquivo in os.listdir(pasta_classe):

        # Cria o caminho completo da imagem
        caminho = os.path.join(

            # Pasta da classe
            pasta_classe,

            # Nome do arquivo
            arquivo
        )

        # Verifica se o caminho corresponde a um arquivo
        if not os.path.isfile(caminho):

            # Ignora o item caso ele não seja um arquivo
            continue

        # Carrega a imagem utilizando o OpenCV
        img = cv2.imread(caminho)

        # Verifica se a imagem foi carregada corretamente
        if img is None:

            # Ignora a imagem caso tenha ocorrido erro na leitura
            continue

        # Converte o padrão de cores BGR para RGB
        img = cv2.cvtColor(

            # Imagem que será convertida
            img,

            # Define a conversão de BGR para RGB
            cv2.COLOR_BGR2RGB
        )

        # Redimensiona a imagem
        img = cv2.resize(

            # Imagem que será redimensionada
            img,

            # Define o tamanho como 128x128
            (IMG_SIZE, IMG_SIZE)
        )

        # Adiciona a imagem processada à lista
        imagens.append(img)

        # Adiciona o rótulo correspondente à imagem
        labels.append(

            # Obtém o número associado à classe
            classe_para_label[classe]
        )


# =====================================================
# CONVERSÃO DOS DADOS
# =====================================================

# Converte a lista de imagens para um array NumPy
X = np.array(

    # Lista contendo todas as imagens
    imagens,

    # Define os dados como números de ponto flutuante
    dtype=np.float32

# Divide os pixels por 255 para normalizar entre 0 e 1
) / 255.0

# Converte os rótulos para um array NumPy
y = np.array(labels)


# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título do dataset
print("DATASET")

# Imprime outro separador
print("========================================")

# Inicia a impressão da quantidade de imagens
print(

    # Texto apresentado na tela
    "Total de imagens:",

    # Obtém a quantidade de imagens
    len(X)
)


# Exibe o título da distribuição das classes
print("\nDistribuição das classes:")

# Percorre as classes e seus respectivos índices
for classe, idx in classe_para_label.items():

    # Exibe a quantidade de imagens da classe
    print(

        # Conta quantos rótulos possuem o índice da classe
        f"{classe}: {np.sum(y == idx)}"
    )


# =====================================================
# TREINO / VALIDAÇÃO / TESTE
# =====================================================

# Divide os dados inicialmente entre treino e conjunto temporário
X_train, X_temp, y_train, y_temp = train_test_split(

    # Imagens que serão divididas
    X,

    # Rótulos que serão divididos
    y,

    # Reserva 30% dos dados para validação e teste
    test_size=0.30,

    # Mantém a proporção das classes
    stratify=y,

    # Mantém a divisão reproduzível
    random_state=SEED
)


# Divide o conjunto temporário entre validação e teste
X_val, X_test, y_val, y_test = train_test_split(

    # Imagens do conjunto temporário
    X_temp,

    # Rótulos do conjunto temporário
    y_temp,

    # Separa metade para teste e metade para validação
    test_size=0.50,

    # Mantém a proporção das classes
    stratify=y_temp,

    # Mantém a divisão reproduzível
    random_state=SEED
)


# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título
print("DIVISÃO DOS DADOS")

# Imprime outro separador
print("========================================")

# Exibe a quantidade de imagens de treinamento
print(

    # Texto apresentado
    "Treino:",

    # Quantidade de imagens de treinamento
    len(X_train)
)

# Exibe a quantidade de imagens de validação
print(

    # Texto apresentado
    "Validação:",

    # Quantidade de imagens de validação
    len(X_val)
)

# Exibe a quantidade de imagens de teste
print(

    # Texto apresentado
    "Teste:",

    # Quantidade de imagens de teste
    len(X_test)
)


# =====================================================
# DATA AUGMENTATION
# =====================================================

# Cria o gerador responsável pelo aumento artificial dos dados
datagen = tf.keras.preprocessing.image.ImageDataGenerator(

    # Permite rotações aleatórias de até 20 graus
    rotation_range=20,

    # Permite deslocamento horizontal de até 15%
    width_shift_range=0.15,

    # Permite deslocamento vertical de até 15%
    height_shift_range=0.15,

    # Permite alterações de zoom de até 15%
    zoom_range=0.15,

    # Define o intervalo de alteração do brilho
    brightness_range=[

        # Brilho mínimo
        0.7,

        # Brilho máximo
        1.3
    ],

    # Permite uma pequena inclinação da imagem
    shear_range=0.10
)


# Ajusta o gerador aos dados de treinamento
datagen.fit(X_train) 


# =====================================================
# MODELO CNN
# =====================================================

# Cria um modelo de rede neural sequencial
model = tf.keras.Sequential([


    # =================================================
    # ENTRADA
    # =================================================

    # Cria a camada de entrada da rede
    tf.keras.layers.Input(

        # Define o formato das imagens de entrada
        shape=(

            # Define a altura da imagem
            IMG_SIZE,

            # Define a largura da imagem
            IMG_SIZE,

            # Define os três canais RGB
            3
        )
    ),


    # =================================================
    # BLOCO 1
    # =================================================

    # Cria a primeira camada convolucional
    tf.keras.layers.Conv2D(

        # Utiliza 32 filtros
        32,

        # Define filtros de tamanho 3x3
        (3, 3),

        # Mantém as dimensões espaciais
        padding='same',

        # Utiliza a função de ativação ReLU
        activation='relu'
    ),

    # Normaliza as ativações da camada anterior
    tf.keras.layers.BatchNormalization(),

    # Reduz as dimensões dos mapas de características
    tf.keras.layers.MaxPooling2D(),

    # Cria uma camada Dropout
    tf.keras.layers.Dropout(

        # Desativa aleatoriamente 20% dos neurônios
        0.20
    ),


    # =================================================
    # BLOCO 2
    # =================================================

    # Cria a segunda camada convolucional
    tf.keras.layers.Conv2D(

        # Utiliza 64 filtros
        64,

        # Define filtros de tamanho 3x3
        (3, 3),

        # Mantém as dimensões espaciais
        padding='same',

        # Utiliza a função ReLU
        activation='relu'
    ),

    # Normaliza as ativações
    tf.keras.layers.BatchNormalization(),

    # Reduz os mapas de características
    tf.keras.layers.MaxPooling2D(),

    # Cria uma camada Dropout
    tf.keras.layers.Dropout(

        # Desativa 25% dos neurônios
        0.25
    ),


    # =================================================
    # BLOCO 3
    # =================================================

    # Cria a terceira camada convolucional
    tf.keras.layers.Conv2D(

        # Utiliza 128 filtros
        128,

        # Define filtros de tamanho 3x3
        (3, 3),

        # Mantém as dimensões espaciais
        padding='same',

        # Utiliza a função ReLU
        activation='relu'
    ),

    # Normaliza as ativações
    tf.keras.layers.BatchNormalization(),

    # Reduz os mapas de características
    tf.keras.layers.MaxPooling2D(),

    # Cria uma camada Dropout
    tf.keras.layers.Dropout(

        # Desativa 30% dos neurônios
        0.30
    ),


    # =================================================
    # BLOCO 4
    # =================================================

    # Cria a quarta camada convolucional
    tf.keras.layers.Conv2D(

        # Utiliza 256 filtros
        256,

        # Define filtros de tamanho 3x3
        (3, 3),

        # Mantém as dimensões espaciais
        padding='same',

        # Utiliza a função ReLU
        activation='relu'
    ),

    # Normaliza as ativações
    tf.keras.layers.BatchNormalization(),

    # Reduz os mapas de características
    tf.keras.layers.MaxPooling2D(),

    # Cria uma camada Dropout
    tf.keras.layers.Dropout(

        # Desativa 35% dos neurônios
        0.35
    ),


    # =================================================
    # CLASSIFICADOR
    # =================================================

    # Calcula a média global dos mapas de características
    tf.keras.layers.GlobalAveragePooling2D(),

    # Cria uma camada totalmente conectada
    tf.keras.layers.Dense(

        # Define 256 neurônios
        256,

        # Utiliza a função ReLU
        activation='relu'
    ),

    # Cria uma camada Dropout
    tf.keras.layers.Dropout(

        # Desativa 50% dos neurônios
        0.50
    ),

    # Cria a camada final de classificação
    tf.keras.layers.Dense(

        # Cria um neurônio para cada classe
        len(classes),

        # Converte os resultados em probabilidades
        activation='softmax'
    )
])


# =====================================================
# COMPILAÇÃO
# =====================================================

# Configura o modelo para o treinamento
model.compile(

    # Define o otimizador Adam
    optimizer=tf.keras.optimizers.Adam(

        # Define a taxa de aprendizagem
        learning_rate=0.0001
    ),

    # Define a função utilizada para calcular o erro
    loss='sparse_categorical_crossentropy',

    # Define as métricas utilizadas durante o treinamento
    metrics=[

        # Utiliza a acurácia
        'accuracy'
    ]
)


# =====================================================
# RESUMO DO MODELO
# =====================================================

# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título
print("MODELO")

# Imprime outro separador
print("========================================")

# Exibe a estrutura completa da rede neural
model.summary()


# =====================================================
# CALLBACKS = é uma função ou objeto que é chamado automaticamente em determinados momentos de um processo.
# =====================================================

# Cria o callback que pode interromper o treinamento
early_stop = tf.keras.callbacks.EarlyStopping(

    # Observa a acurácia da validação
    monitor='val_accuracy',

    # Espera até 10 épocas sem melhora
    patience=10,

    # Recupera os pesos da melhor época
    restore_best_weights=True,

    # Exibe informações durante a execução
    verbose=1
)


# Cria o callback para reduzir a taxa de aprendizagem
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    # Observa o erro da validação
    monitor='val_loss',

    # Divide a taxa de aprendizagem pela metade
    factor=0.5,

    # Espera três épocas sem melhora
    patience=3,

    # Define a menor taxa de aprendizagem permitida
    min_lr=1e-7,

    # Exibe informações durante a execução
    verbose=1
)


# Cria o callback responsável por salvar o melhor modelo
checkpoint = tf.keras.callbacks.ModelCheckpoint(

    # Define o nome do arquivo
    'melhor_modelo.keras',

    # Observa a acurácia da validação
    monitor='val_accuracy',

    # Salva somente quando houver melhora
    save_best_only=True,

    # Exibe informações quando o modelo for salvo
    verbose=1
)


# =====================================================
# TREINAMENTO
# =====================================================

# Pula uma linha e imprime um separador
print("\n========================================")

# Informa o início do treinamento
print("INICIANDO TREINAMENTO")

# Imprime outro separador
print("========================================")


# Verifica se existe uma GPU disponível
if gpus:

    # Informa que a GPU será utilizada
    print("Dispositivo utilizado: GPU")

    # Define explicitamente a primeira GPU para o treinamento
    with tf.device('/GPU:0'):

        # Inicia o treinamento e armazena seu histórico
        history = model.fit(

            # Cria os lotes de treinamento com Data Augmentation
            datagen.flow(

                # Fornece as imagens de treinamento
                X_train,

                # Fornece os rótulos de treinamento
                y_train,

                # Define 32 imagens por lote
                batch_size=32
            ),

            # Define no máximo 60 épocas de treinamento
            epochs=60,

            # Define os dados utilizados para validação
            validation_data=(

                # Imagens de validação
                X_val,

                # Rótulos de validação
                y_val
            ),

            # Define os callbacks utilizados no treinamento
            callbacks=[

                # Callback de parada antecipada
                early_stop,

                # Callback de redução da taxa de aprendizagem
                reduce_lr,

                # Callback que salva o melhor modelo
                checkpoint
            ],

            # Exibe o progresso do treinamento
            verbose=1
        )

# Executado quando nenhuma GPU está disponível
else:

    # Informa que a CPU será utilizada
    print("Dispositivo utilizado: CPU")

    # Inicia o treinamento e armazena o histórico
    history = model.fit(

        # Cria os lotes com Data Augmentation
        datagen.flow(

            # Fornece as imagens de treinamento
            X_train,

            # Fornece os rótulos de treinamento
            y_train,

            # Define 32 imagens por lote
            batch_size=32
        ),

        # Define no máximo 60 épocas
        epochs=60,

        # Define os dados utilizados para validação
        validation_data=(

            # Imagens de validação
            X_val,

            # Rótulos de validação
            y_val
        ),

        # Define os callbacks utilizados
        callbacks=[

            # Callback de parada antecipada
            early_stop,

            # Callback de redução da taxa de aprendizagem
            reduce_lr,

            # Callback que salva o melhor modelo
            checkpoint
        ],

        # Exibe o progresso do treinamento
        verbose=1
    )


# =====================================================
# AVALIAÇÃO FINAL
# =====================================================

# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título da avaliação
print("AVALIAÇÃO FINAL")

# Imprime outro separador
print("========================================")


# Avalia o modelo utilizando os dados de teste
loss, acc = model.evaluate(

    # Fornece as imagens de teste
    X_test,

    # Fornece os rótulos corretos
    y_test,

    # Não exibe a barra de progresso
    verbose=0
)


# Exibe a acurácia final do modelo
print(

    # Mostra a acurácia com quatro casas decimais
    f"Accuracy Final: {acc:.4f}"
)


# =====================================================
# PREDIÇÕES
# =====================================================

# Informa que as predições serão geradas
print("\nGerando predições...")

# Realiza as previsões utilizando as imagens de teste
prob = model.predict(

    # Fornece as imagens de teste
    X_test
)

# Obtém a classe com maior probabilidade
pred = np.argmax(

    # Probabilidades calculadas pelo modelo
    prob,

    # Procura o maior valor entre as classes
    axis=1
)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título do relatório
print("CLASSIFICATION REPORT")

# Imprime outro separador
print("========================================\n")


# Exibe o relatório completo de classificação
print(

    # Gera o relatório de classificação
    classification_report(

        # Fornece as classes verdadeiras
        y_test,

        # Fornece as classes previstas
        pred,

        # Define os nomes das classes
        target_names=classes
    )
)


# =====================================================
# MATRIZ DE CONFUSÃO
# =====================================================

# Pula uma linha e imprime um separador
print("\n========================================")

# Exibe o título
print("MATRIZ DE CONFUSÃO")

# Imprime outro separador
print("========================================\n")


# Exibe a matriz de confusão
print(

    # Calcula a matriz de confusão
    confusion_matrix(

        # Fornece as classes verdadeiras
        y_test,

        # Fornece as classes previstas
        pred
    )
)


# =====================================================
# SALVAR MODELO
# =====================================================

# Salva o modelo treinado em um arquivo Keras
model.save(

    # Define o nome do arquivo do modelo
    "modelo_libras.keras"
)


# Salva as classes utilizadas pelo modelo
np.save(

    # Define o nome do arquivo
    "classes.npy",

    # Converte a lista de classes para um array NumPy
    np.array(classes)
)


# =====================================================
# FINAL
# =====================================================

# Pula uma linha e imprime um separador
print("\n========================================")

# Informa que o treinamento terminou
print("TREINAMENTO FINALIZADO")

# Imprime outro separador
print("========================================")

# Informa onde o modelo foi salvo
print(

    # Nome do arquivo do modelo
    "Modelo salvo: modelo_libras.keras"
)

# Informa onde as classes foram salvas
print(

    # Nome do arquivo das classes
    "Classes salvas: classes.npy"
)

# Exibe a lista de classes utilizadas
print(

    # Texto apresentado antes das classes
    "Classes:",

    # Lista contendo todas as classes
    classes
)