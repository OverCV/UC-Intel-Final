Cuándo usar cada una
de antes muy bien Cuándo debes usar cada
función depende de algunas cosas la
función logística o sigmoid se
recomienda solamente en la capa de
salida cuando requerimos un resultado
binario cero o uno perro o gato calvo o
con pelo la tangente hiperbólica o tan
es superior a la función logística para
capas ocultas sin embargo en muy pocos
casos da resultados superiores a relo
por ejemplo en redes muy simples de
regresión relu es la que normalmente vas
a utilizar A menos que quieras
experimentar mucho con otras funciones
si te topas mucho con problemas como las
neuronas muertas puedes probar
disminuyendo la tasa de aprendizaje o
explorando con otras funciones el rel y
p relu puedes explorarlas si batallas
mucho con las neuronas muertas de relu
Sin embargo estas pueden caer un poco en
el problema de desvanecimiento de
gradiente gelu Aunque tiene tiempo que
se propuso y es utilizada en redes
famosas como gpt y otros Transformers
aún en muchas publicaciones es
considerada como relativamente nueva de
manera general tiene mejores resultados
que relu especialmente en redes muy
grandes o Transformers puedes
experimentar con ella y comparar los
resultados contra relu switch Funciona
muy bien y en general vence arru Solo
que Su costo computacional es un poco
más alto Mich en tareas de visión
computacional ha dado mejores resultados
incluso que switch y de hecho se utilizó
en Yolo 4 pero en Yolo 5 fue reemplazado
por el relu y
sigmoid utiliza softmax en la última
capa de tus redes de clasificación la
función de identidad lineal o no
utilizar una función de activación
puedes hacerlo en tus redes cuando son
de regresión ahora hay otras funciones
