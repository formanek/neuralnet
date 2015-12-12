package cz.muni.fi.pv021.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.DoubleSupplier;

/**
 * Artificial neural network with neurons organized into layers (fully connected)
 *
 * @author David Formanek
 */
public class LayeredNeuralNetwork {

    private final InputNeuralNetworkLayer inputLayer;
    private final List<NeuralNetworkLayer> layers;

    /**
     * Constructs network with input layer only, other layers must be added
     *
     * @param inputLayer constructed input layer of the network
     */
    public LayeredNeuralNetwork(InputNeuralNetworkLayer inputLayer) {
        Objects.requireNonNull(inputLayer, "input layer null");
        this.inputLayer = inputLayer;
        layers = new ArrayList<>();
    }

    /**
     * Gets the input layer used in this network
     *
     * @return the input layer
     */
    public InputNeuralNetworkLayer getInputLayer() {
        return inputLayer;
    }

    /**
     * Gets hidden layers and the output layer as the last in the list
     *
     * @return all layers except from the input layer
     */
    public List<NeuralNetworkLayer> getLayers() {
        return layers;
    }

    /**
     * Adds the layer into network and connects it with the layer below, last call adds output layer
     *
     * @param layer layer to add at the top
     * @param initialWeightSupplier function to specify initial weights (before learning)
     */
    public void addLayer(NeuralNetworkLayer layer, DoubleSupplier initialWeightSupplier) {
        Objects.requireNonNull(layer, "adding null layer");
        Objects.requireNonNull(initialWeightSupplier, "adding null initial weight supplier");
        if (layers.isEmpty()) {
            inputLayer.connectTo(layer, initialWeightSupplier);
        } else {
            layers.get(layers.size() - 1).connectTo(layer, initialWeightSupplier);
        }
        layers.add(layer);
    }

    /**
     * Assigns input data to neurons in the input layer
     *
     * @param input double value for each input neuron
     */
    public void setData(double[] input) {
        Objects.requireNonNull(input, "setting null data");
        List<InputNeuron> neurons = inputLayer.getNeurons();
        int size = neurons.size();
        if (size != input.length) {
            throw new IllegalArgumentException("input size does not match number of input neurons");
        }
        for (int i = 0; i < size; i++) {
            neurons.get(i).setValue(input[i]);
        }
    }

    /**
     * Computes functions in all neurons (gradually in layers) and gets the output layer result
     *
     * @return the result of computations in output neurons (propagated from the whole network)
     */
    public double[] getResult() {
        if (layers.isEmpty()) {
            throw new IllegalStateException("no output layer exists");
        }
        for (NeuralNetworkLayer layer : layers) {
            layer.computeFunctions();
        }
        List<Neuron> outputNeurons = layers.get(layers.size() - 1).getNeurons();
        int size = outputNeurons.size();
        double[] outputs = new double[size];
        for (int i = 0; i < size; i++) {
            outputs[i] = outputNeurons.get(i).getFunctionOutput();
        }
        return outputs;
    }
}
