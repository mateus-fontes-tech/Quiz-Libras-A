# Quiz Libras

Aplicação web para reconhecimento de sinais do alfabeto em Libras por meio de inteligência artificial e câmera.

Documentação técnica baseada no arquivo `train_libras.py` fornecido pelo autor. Descreve a implementação enviada; o treinamento não foi executado para produzir este documento e não foram fornecidos resultados numéricos.

## 1. Finalidade

O script carrega imagens organizadas por classe, treina uma rede neural convolucional (CNN), avalia as previsões em um conjunto de teste e salva o modelo e seus rótulos. Esses arquivos podem ser utilizados posteriormente pelo Quiz Libras.

Este arquivo é um programa de treinamento. Ele não contém servidor Flask, interface web, captura de câmera ou rotas de predição. O reconhecimento de classes de imagens do alfabeto manual também não equivale à tradução completa de Libras.

## 2. Como organizar a documentação

Coloque este `README.md` na mesma pasta do script. Ele pode ser lido no editor ou exibido no repositório. Os comentários existentes no Python explicam instruções individuais; este documento apresenta o funcionamento completo.

Opcionalmente, acrescente este resumo no início do Python, antes dos imports:

```python
"""Treinamento de uma CNN para classificação de imagens de Libras.

Entrada: imagens em subpastas por classe, no caminho DIRETORIO.
Processamento: RGB, 128 x 128 pixels e normalização por 255.
Divisão: aproximadamente 70% treino, 15% validação e 15% teste.
Saídas: melhor_modelo.keras, modelo_libras.keras e classes.npy.

Consulte README.md para arquitetura, parâmetros e execução.
"""
```

Não é necessário importar o README nem acrescentar sua leitura ao programa.

## 3. Entradas e organização das imagens

`DIRETORIO` aponta para a pasta que contém as subpastas das classes. No arquivo enviado, está configurado assim:

```python
DIRETORIO = r'C:\programacao\Python\redes_neorais\treinamento_IA\imagens'
```

Altere esse valor conforme o local das imagens em seu computador. No Linux ou WSL, use um caminho acessível nesse ambiente.

Exemplos de organização:

| Caminho ilustrativo | Significado |
| --- | --- |
| `imagens/A/exemplo01.jpg` | Imagem rotulada como A |
| `imagens/B/exemplo01.jpg` | Imagem rotulada como B |
| `imagens/C/exemplo01.jpg` | Imagem rotulada como C |

Os nomes acima são exemplos. O script descobre as classes pelas subpastas existentes, ordena seus nomes e atribui índices a partir de zero. Não há quantidade fixa de classes definida no código.

A leitura abrange os arquivos diretamente dentro de cada subpasta de classe. Subpastas adicionais não são percorridas. Arquivos que o OpenCV não consegue ler são ignorados, sem relatório individual do descarte.

## 4. Dependências

| Biblioteca | Utilização no script |
| --- | --- |
| OpenCV (`cv2`) | Leitura, conversão de cores e redimensionamento |
| NumPy | Arrays, normalização, rótulos e gravação de classes |
| TensorFlow / Keras | GPU, camadas, aumento de dados, treino e salvamento |
| scikit-learn | Divisão dos dados e métricas de classificação |
| `os` | Acesso a diretórios; faz parte do Python |

O arquivo enviado não especifica versões das bibliotecas. Portanto, esta documentação não estabelece uma combinação de versões testada nem garante compatibilidade com qualquer instalação. Registre o ambiente que efetivamente executar o treinamento.

## 5. Etapas do processamento

### 5.1 Detecção da GPU

`tf.config.list_physical_devices('GPU')` consulta as GPUs reconhecidas pelo TensorFlow. Quando existem, o programa tenta ativar o crescimento gradual da alocação de memória por meio de `set_memory_growth`. Isso não limita o consumo a um valor fixo.

Na etapa de treinamento, o código utiliza o contexto `tf.device('/GPU:0')` quando detecta GPU. Caso contrário, segue o ramo de CPU. A configuração de GPU não elimina a necessidade de RAM para carregar o dataset.

### 5.2 Sementes aleatórias

`SEED = 42` é utilizado pelo NumPy, pelo TensorFlow e nas divisões dos dados. Isso auxilia a repetibilidade, mas não garante resultados idênticos entre ambientes. A leitura dos arquivos dentro de cada classe usa `os.listdir` sem ordenação; mudanças nessa ordem também podem alterar quais imagens participam de cada conjunto.

### 5.3 Leitura e pré-processamento

Para cada imagem válida, o script:

1. Carrega o arquivo com `cv2.imread`.
2. Converte a ordem dos canais de BGR para RGB.
3. Redimensiona para 128 × 128 pixels.
4. Guarda a imagem e o índice da classe em listas.
5. Converte as imagens para um array `float32` e divide seus valores por 255.

Antes do aumento de dados, `X` contém valores entre 0 e 1, com formato `(N, 128, 128, 3)`. `N` corresponde ao total de imagens válidas. `y` contém um rótulo inteiro para cada imagem. O código utiliza imagens coloridas, não escala de cinza.

### 5.4 Divisão dos dados

| Conjunto | Proporção aproximada | Finalidade |
| --- | --- | --- |
| Treinamento | 70% | Ajustar os parâmetros da rede |
| Validação | 15% | Acompanhar o treino e orientar os callbacks |
| Teste | 15% | Avaliação final após o treinamento |

A primeira divisão separa 30% dos dados em um conjunto temporário. A segunda divide esse conjunto ao meio. As quantidades exatas dependem dos arredondamentos e do número de imagens.

As duas divisões usam estratificação para preservar aproximadamente a distribuição das classes. Classes com poucas imagens podem inviabilizar essa divisão.

### 5.5 Aumento de dados

O `ImageDataGenerator` cria transformações durante a geração dos lotes de treinamento. Essas variações não são gravadas como novas imagens no disco.

| Configuração | Valor no código |
| --- | --- |
| Rotação | `rotation_range=20` |
| Deslocamento horizontal | `width_shift_range=0.15` |
| Deslocamento vertical | `height_shift_range=0.15` |
| Zoom | `zoom_range=0.15` |
| Brilho | `brightness_range=[0.7, 1.3]` |
| Cisalhamento | `shear_range=0.10` |

O objetivo é apresentar variações dos exemplos à rede. É necessário conferir visualmente se as transformações preservam a classe da mão. Validação e teste são fornecidos diretamente, sem esse gerador.

O script chama `datagen.fit(X_train)`. Não há configuração de normalização por estatísticas do dataset ou ZCA nesta instância; essa chamada não representa treinamento da CNN.

## 6. Arquitetura da CNN

Cada bloco convolucional combina extração de características, normalização das ativações, redução espacial e Dropout.

| Etapa | Componentes | Formato de saída por imagem |
| --- | --- | --- |
| Entrada | Imagem RGB | 128 × 128 × 3 |
| Bloco 1 | Conv2D: 32 filtros 3 × 3, ReLU, `same`; BatchNormalization; MaxPooling2D; Dropout 0,20 | 64 × 64 × 32 |
| Bloco 2 | Conv2D: 64 filtros 3 × 3, ReLU, `same`; BatchNormalization; MaxPooling2D; Dropout 0,25 | 32 × 32 × 64 |
| Bloco 3 | Conv2D: 128 filtros 3 × 3, ReLU, `same`; BatchNormalization; MaxPooling2D; Dropout 0,30 | 16 × 16 × 128 |
| Bloco 4 | Conv2D: 256 filtros 3 × 3, ReLU, `same`; BatchNormalization; MaxPooling2D; Dropout 0,35 | 8 × 8 × 256 |
| Agregação | GlobalAveragePooling2D | 256 valores |
| Classificador | Dense: 256 unidades, ReLU; Dropout 0,50 | 256 valores |
| Saída | Dense: `len(classes)` unidades, Softmax | Uma pontuação por classe |

**Conv2D** aprende filtros para padrões visuais. **BatchNormalization** normaliza ativações e utiliza estatísticas acumuladas na inferência. **MaxPooling2D** reduz as dimensões espaciais. **Dropout** zera aleatoriamente parte das ativações durante o treinamento, como forma de regularização.

**GlobalAveragePooling2D** calcula uma média por mapa de características. Neste código, não há camada Flatten. **Dense** combina as características, e **Softmax** produz uma distribuição sobre as classes; essa distribuição não garante que a previsão esteja correta.

## 7. Configuração do treinamento

| Parâmetro | Valor |
| --- | --- |
| Otimizador | Adam |
| Taxa de aprendizagem inicial | 0,0001 |
| Função de perda | `sparse_categorical_crossentropy` |
| Métrica acompanhada | `accuracy` |
| Tamanho do lote | 32 |
| Máximo de épocas | 60 |

A função de perda utiliza os rótulos inteiros; o código não converte os rótulos em vetores one-hot. Uma época corresponde a uma passagem pelos lotes de treinamento. O treino pode terminar antes de 60 épocas por ação da parada antecipada.

### Callbacks

| Callback | Configuração | Comportamento |
| --- | --- | --- |
| EarlyStopping | `val_accuracy`, paciência 10, `restore_best_weights=True` | Pode interromper o treino após 10 épocas sem melhora e está configurado para recuperar os melhores pesos |
| ReduceLROnPlateau | `val_loss`, fator 0,5, paciência 3, mínimo `1e-7` | Reduz a taxa de aprendizagem quando a perda de validação deixa de melhorar |
| ModelCheckpoint | `val_accuracy`, `save_best_only=True` | Salva `melhor_modelo.keras` quando há melhora na métrica monitorada |

A variável `history` recebe o histórico retornado pelo treinamento, mas não é exportada em arquivo pelo script.

## 8. Avaliação

O código avalia o modelo em memória com `X_test` e `y_test`, calcula previsões e seleciona a classe de maior saída com `np.argmax(..., axis=1)`.

| Saída | Interpretação |
| --- | --- |
| `loss` | Perda no teste; retornada e armazenada, mas não impressa explicitamente |
| `Accuracy Final` | Proporção de acertos no teste, impressa com quatro casas decimais |
| Precisão | Entre previsões de uma classe, proporção correta |
| Revocação / recall | Entre exemplos reais de uma classe, proporção identificada |
| F1-score | Média harmônica entre precisão e revocação |
| Support | Quantidade de exemplos reais de cada classe no teste |
| Matriz de confusão | Linhas representam classes reais; colunas representam previsões |

O relatório e a matriz são impressos no terminal. O script não gera gráficos, imagens da matriz ou arquivos CSV. Não há resultados de execução anexados que permitam informar a acurácia alcançada.

## 9. Arquivos produzidos

| Arquivo | Momento e finalidade |
| --- | --- |
| `melhor_modelo.keras` | Criado ou atualizado pelo checkpoint durante o treino |
| `modelo_libras.keras` | Salva o modelo em memória após a avaliação final |
| `classes.npy` | Salva a lista ordenada de classes associada às saídas da rede |

Os caminhos de saída são relativos ao diretório de trabalho do processo. Assim, não são necessariamente a pasta onde o script está salvo. Uma nova execução pode sobrescrever arquivos com esses nomes.

O código não recarrega `melhor_modelo.keras` antes da avaliação. Avalia e salva o objeto `model` em memória, com o comportamento de restauração configurado no EarlyStopping.

## 10. Execução

1. Disponibilize as bibliotecas importadas no ambiente Python do projeto.
2. Organize as imagens em subpastas por classe e ajuste `DIRETORIO`.
3. Abra o terminal na pasta em que deseja gerar os modelos.
4. Execute o script com seu nome real. Mantendo o nome enviado:

```bash
python "Código colado(8).py"
```

5. Acompanhe a detecção da GPU, as contagens por classe, o treinamento e os resultados finais.
6. Preserve o modelo e seu `classes.npy` correspondente para integração à aplicação.

O script executa as operações no nível principal do módulo. Importá-lo em outro programa também inicia seu processamento. Por isso, a aplicação web deve carregar o modelo salvo, em vez de importar este script de treinamento.

## 11. Integração com o Quiz Libras

O backend do quiz deve carregar `modelo_libras.keras` e o `classes.npy` da mesma execução. Para uma nova imagem, deve reproduzir o pré-processamento: RGB, 128 × 128, `float32` e divisão por 255. Antes da predição, deve adicionar a dimensão do lote, resultando em `(1, 128, 128, 3)`.

A posição de maior saída do modelo deve ser interpretada usando a lista salva, sem reordenar seus elementos. Não se deve executar as transformações aleatórias de treinamento para uma predição comum.

Esta integração descreve o contrato de entrada e saída necessário. O arquivo recebido não contém o backend do quiz, portanto sua implementação não foi conferida.

## 12. Cuidados e limitações da implementação

- **Memória:** todas as imagens são carregadas na RAM. Somente o array normalizado exige `N × 128 × 128 × 3 × 4` bytes; listas e cópias das divisões elevam esse consumo. O crescimento gradual da memória da GPU não resolve essa demanda de RAM.
- **Qualidade das classes:** pastas vazias, poucos exemplos e arquivos inválidos podem impedir a divisão ou causar falhas posteriores. Não há validação explícita para essas situações.
- **Separação por imagem:** o código não separa participantes, sessões ou origens. Imagens duplicadas ou muito parecidas em conjuntos diferentes podem tornar a avaliação otimista. Quando houver esses agrupamentos, considere-os na separação dos dados.
- **Aumento de dados:** confira um lote gerado, inclusive sua faixa numérica, para verificar a interação entre a normalização prévia e a transformação de brilho na versão utilizada.
- **Generalização:** bons resultados em teste controlado não garantem a mesma qualidade em imagens com outros fundos, iluminação e pessoas.
- **Reprodutibilidade:** registre versões, origem do dataset, classes, contagens, divisões e resultados de cada execução. As sementes ajudam, mas não documentam sozinhas o experimento.

## 13. Resumo metodológico para o trabalho

O processamento inicia com a leitura de imagens organizadas em diretórios por classe. As imagens válidas são convertidas de BGR para RGB, redimensionadas para 128 × 128 pixels e normalizadas pela divisão dos valores por 255. Os dados são separados de forma estratificada em aproximadamente 70% para treinamento, 15% para validação e 15% para teste. Durante o treinamento, são aplicadas transformações de rotação, deslocamento, zoom, brilho e cisalhamento. A arquitetura utiliza quatro blocos convolucionais com 32, 64, 128 e 256 filtros, seguidos de agregação global por média e camadas densas. O treinamento emprega Adam, taxa inicial de 0,0001, lotes de 32 imagens e limite de 60 épocas, com parada antecipada, redução da taxa de aprendizagem e salvamento do melhor modelo segundo a acurácia de validação. A etapa final prevê avaliação por acurácia, relatório de classificação e matriz de confusão no conjunto de teste, seguida da gravação do modelo e dos rótulos das classes.

Este resumo descreve o procedimento implementado; resultados e conclusões experimentais dependem dos registros da execução.
# Quiz-Libras

