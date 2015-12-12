package cz.muni.fi.pv021.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * Input layer for layered network, consists of input neurons
 *
 * @author David Formanek
 */
public class InputNeuralNetworkLayer {

    private final List<InputNeuron> neurons;

    /**
     * Constructs input layer with given number of input neurons (without bias)
     *
     * @param size number of input neurons
     */
    public InputNeuralNetworkLayer(int size) {
        if (size < 1) {
            throw new IllegalArgumentException("layer size must be at least 1");
        }
        neurons = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            neurons.add(new InputNeuron());
        }
    }

    /**
     * Gets neurons created by this layer
     *
     * @return all input neurons
     */
    public List<InputNeuron> getNeurons() {
        return neurons;
    }

    /**
     * Connects the input layer with first hidden layer (or output layer if there are only 2 layers)
     *
     * @param layer the second layer (first non-input) of the constructed network
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
}
