NEURAL Lib
==========

Neural es una librería que implementa métodos y modelos de Redes Neuronales Artificiales (RNA) en python.  Su diseño está especialmente pensado para el uso en ámbitos académicos y de investigación y surge de la necesidad de una librería que permita crear fácilmente RNA de los modelos más conocidos como así también experimentar con gran variedad de combinaciones de distintas técnicas.


Objetivo
========

El principal interés de esta librería es brindar una forma práctica de trabajo con la mayoría de los modelos de RNA conocidos manteniendo una interfaz consistente para todos y objetos que auto-contengan los métodos necesarios para la funcionalidad del modelo.  Pero también se busca facilitar la experimentación con nuevos modelos que combinen distintos tipos de conexiones entre capas, funciones de activación, reglas de aprendizaje y métodos de entrenamiento.
Para poder lograr todo esto la filosofía del diseño se basa en tres puntos:

* Modularidad
* Homogeneidad
* Flexibilidad

Cada una de estas está intrínsecamente relacionadas con las demás, sin embargo es posible pensar en el objetivo individual de cada una.


Modularidad
-----------

Esto consiste en crear componentes que contengan distintas variantes en alguna funcionalidad específica pero común a varios modelos.  A su vez, estos componentes se integrarán con otros componentes más complejos uniendo de forma consistente las funcionalidades de todos.
Esto permite crear cualquier modelo de RNA, conocido o nuevo, a partir de elementos sencillos.


Homogeneidad
------------

Esto consiste en mantener una interfaz uniforme para todos los modelos, independientemente de cuál sea su funcionalidad.  Esto además significa que toda la inteligencia necesaria para el funcionamiento y entrenamiento están auto-contenidas en el modelo.


Flexibilidad
------------

Esto consiste en no limitar las opciones en la creación de nuevos modelos de RNA eligiendo un comportamiento predeterminado por ser muy común o por alguna ventaja de implementación.  Pero también significa ofrecer plantillas o pre-configuraciones que faciliten el uso de los elementos más comunes.



Estructura del paquete
======================

Para entender mejor la organización general de estos elementos se puede ver en el árbol de la estructura del paquete la relación entre todos sus módulos.

```
    neural/
        connection/
            __init__.py
            connection.py
                connection( creation, affinity, rectifier)
                    weights( N: int, M: int): array
                    rectify( W: array): array
                full: connection
                single: connection
                receptive_field: connection
        excitation/
            __init__.py
            excitation.py
                excitation()
                    forw( X: array, W: array): array
                    back( W: array, Y: array): array
                prod: excitation        # Dot Product
                eucl: excitation        # Euclidian Distance
                sad: excitation         # Sum of Absolute Differences
        activation/
            __init__.py
            activation.py
                activation( cod: (real, real), steepness: real)
                    function( X: array): array
                    derivate( X: array): array
                linear: activation
                ramp: activation
                step: activation
                sigmoid: activation
                logistic: activation
                radial: activation
                softmax: activation
                wta: activation
        correction/
            __init__.py
            correction.py
                correction()
                    gradient( feed: neural.feed): array
                    updating( feed: neural.feed): array
                hebbian: correction
                perceptron: correction
                delta: correction
                som: correction
                ojam: correction
                sanger: correction
                instar: correction
                outstar: correction
        estimation/
            __init__.py
            estimation.py
                estimation()
                    cost( layer: neural.layer): real
                sse: estimation     # Sum of Squared Errors
                mse: estimation     # Mean of Squared Errors
                cl: estimation      # Classification Loss
                ce: estimation      # Cross Entropy
                em: estimation      # Entropic Measure
                en: estimation      # Energy
        training/
            __init__.py
            training.py
                training()
                batch: training
                incremental: training
        learning/
            __init__.py
            learning.py
                learning()
                static: learning
                supervised: learning
                unsupervised: learning
                reinforcement: learning
                deep: learning
        model/
        __init__.py
        network.py
            network( topology: neural.topology, teacher: neural:teacher)
        topology.py
            topology( dims: list, con: neural.connection, exc: neural.excitation, act: neural.activation, cor: neural.correction, est: neural.estimation)
        layer.py
            layer( size: int, unit: neural.unit)
        feed.py
            feed( ilayer: neural.layer, olayer: neural.layer, con: neural.connection, exc: neural.excitation)
        unit.py
            unit( act: neural.activation, cor: neural.correction, est: neural.estimation)
        teacher.py
            teacher( training: neural.training, learning: neural.learning, monitor: neural.monitor)
        monitor.py
            monitor()
            null: monitor
            plot: monitor
            prnt: monitor
            grph: monitor
```

* Faltan: trial, dataset, case, pattern, parameter


Modos de uso
============

Pero quizás la mejor forma de entenderlo sea a partir de algunos ejemplos en su modo de uso.
Por ejemplo, para crear una red de modo explícito en un solo paso:

    import neural

    net = neural.network(
            neural.topology( [2,3,1],
                neural.connection.full(),
                neural.excitation.sad(),
                neural.activation.step(),
                neural.correction.hebb(),
                neural.estimation.mme()
            ),
            neural.teacher(
                neural.training.batch(),
                neural.learning.backpropagation(),
                neural.monitor.plot(),
            )
        )

Para crear o agregar elementos a una red:
 
    feed = neural.feed(
                neural.connection.full( min=-0.1, max=0.1),
                neural.excitation.prod()
                )

    unit = neural.unit(
                neural.activation.sigmoid( cod=neural.coding.binary),
                neural.correction.delta( lr=0.1),
                neural.estimation.mse()
                )

    net = neural.network()
    il = net.layer_add( 2, neural.model.unit.input() )
    hl = net.layer_add( 3, unit)
    ol = net.layer_add( 1, unit)
    net.feed_add( il, hl, feed)
    net.feed_add( hl, ol, feed)


O a más bajo nivel:

    net.topology.layer_add( neural.layer( 3, neural.model.unit.perceptron())


Para cargar una red previamente guardada:

    net = neural.network()
    net.load( 'perc-2-3-1.net')


Para crear un modelo de red predeterminado:

    net = neural.model.mlp( [2,3,1])


Para modificar elementos de una red con otros propios:

    class square( neural.activation):
        def function( _, X):
            return X**2

        def derivate( _, X):
            return 2*X

    class experimental_unit( neural.model.unit.mcculloch_pitts):
        def __init__( _):
            neural.model.unit.mcculloch_pitts.__init__( _)
            _.activation = square()

    net = neural.model.mlp( [2,3,1] )
    net.topology.layers[1].unit = experimental_unit()


Implementación
==============

A pesar de que se busca unificar a la mayoría de los modelos de RNA en una sola librería bajo una misma interfaz, la gran variedad de modelos existentes demanda ciertas decisiones de compromiso que inevitablemente dejarán a algunos modelos fuera.

Una de estas decisiones es la implementación de modelos en **capas**. Si bien no todos los modelos están organizados de esta forma la gran mayoría sí lo están.  Las capas contienen las funcionalidades necesarias para el calculo de la activación de las unidades, de la corrección de los pesos y otros, y guardan en vectores algunos de estos.

Es importante notar que la clase **unit** no modela una unidad en particular, sino que modela un **tipo** de unidad, es decir, describe la funcionalidad de un modelo de unidad pero no contiene ni el valor de activación ni ningún otro en particular.

Para la modularización de funcionalidades se eligió usar una combinación de los patrones de diseño [Bridge](http://en.wikipedia.org/wiki/Bridge_pattern), [Decorator](http://en.wikipedia.org/wiki/Decorator_pattern), [Strategy](http://en.wikipedia.org/wiki/Strategy_pattern) y [Template Method](http://en.wikipedia.org/wiki/Template_method_pattern) en donde las funcionalidades de las subclases no son adquiridas por herencia múltiple de las clases padre sino que el constructor de la subclase recibe objetos con las funcionalidades necesarias.
Esto permite crear objetos con nuevas combinaciones de funcionalidad sin necesidad de derivar nuevas clases y además permite reemplazar funcionalidades en tiempo de ejecución.

La librería tiene como dependencia únicamente a [NumPy](http://www.numpy.org/) y opcionalmente a [matplotlib](http://matplotlib.org/) y [VPython](http://vpython.org/).


Licencia
========

[GNU General Public License 3.0](http://www.gnu.org/licenses/gpl-3.0.txt).


<c>
Neural - Copyright (C) 2014
@gustavx 
@FlyingPumba
</c>
