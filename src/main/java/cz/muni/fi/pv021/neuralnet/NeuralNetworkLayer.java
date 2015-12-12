package cz.muni.fi.pv021.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleSupplier;
import java.util.function.UnaryOperator;

/**
 * Standard (non-input) layer of neural network (including the output layer)
 *
 * @author David Formanek
 */
public class NeuralNetworkLayer {

    private final List<Neuron> neurons;

    /**
     * Constructs layer with given number of neurons
     *
     * @param size number of neurons in this layer
     * @param activationFunction function to be applied to scalar product of input neurons outputs
     * @param initialBiasSupplier function to specify initial bias for each neuron
     */
    public NeuralNetworkLayer(int size, UnaryOperator<Double> activationFunction,
            DoubleSupplier initialBiasSupplier) {
        Objects.requireNonNull(activationFunction, "null activation function");
        Objects.requireNonNull(initialBiasSupplier, "null initial bias supplier");
        if (size < 1) {
            throw new IllegalArgumentException("size must be at least 1");
        }
        neurons = new ArrayList<>(size);
        BiasNeuron bias = new BiasNeuron();
        for (int i = 0; i < size; i++) {
            Neuron neuron = new BasicNeuron(activationFunction);
            bias.connectTo(neuron, initialBiasSupplier.getAsDouble());
            neurons.add(neuron);
        }
    }

    /**
     * Gets neurons created by this layer
     *
     * @return all input neurons
     */
    public List<Neuron> getNeurons() {
        return neurons;
    }

    /**
     * Connects the layer to the upper part of the network (and lower part must connect to this)
     *
     * @param layer upper hidden layer (or output layer if this the last hidden layer)
     * @param initialWeigthSupplier function to specify initial weights (before learning)
     */
    public void connectTo(NeuralNetworkLayer layer, DoubleSupplier initialWeigthSupplier) {
        Objects.requireNonNull(layer, "null layer");
        Objects.requireNonNull(initialWeigthSupplier, "null initial weight supplier");
        for (Neuron fromNeuron : neurons) {
            for (Neuron toNeuron : layer.getNeurons()) {
                fromNeuron.connectTo(toNeuron, initialWeigthSupplier.getAsDouble());
            }
        }
    }

    /**
     * Computes activation functions in each neuron in this layer
     */
    public void computeFunctions() {
        for (Neuron neuron : neurons) {
            neuron.computeFunction();
        }
    }
}
