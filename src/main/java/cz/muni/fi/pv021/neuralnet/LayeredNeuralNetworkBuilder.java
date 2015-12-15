package cz.muni.fi.pv021.neuralnet;

import java.util.Objects;
import java.util.function.DoubleSupplier;
import java.util.function.UnaryOperator;

/**
 * This class simplifies building a layered neural network
 *
 * Layers are created automatically according to specified architecture, all hidden layers use the
 * same given activation function (output layer uses just identity) and the same given initial
 * weight supplier is used to initialize all the weights and biases
 *
 * @author David Formanek
 */
public class LayeredNeuralNetworkBuilder {

    private int[] architecture;
    private UnaryOperator<Double> activationFunction;
    private UnaryOperator<Double> activationFunctionDerivation;
    private DoubleSupplier initialWeigthSupplier;

    /**
     * Sets the network depth (number of parameters) and width of each layer
     *
     * @param architecture numbers of neurons in layers (from input to output)
     */
    public void setArchitecture(int... architecture) {
        Objects.requireNonNull(architecture, "setting null architecture");
        if (architecture.length < 2) {
            throw new IllegalArgumentException("At least 2 layers needed (1 is the input)");
        }
        this.architecture = architecture;
    }

    /**
     * Sets the activation function for all neurons in hidden layers
     *
     * @param activationFunction function to be applied to scalar product of input neurons outputs
     */
    public void setActivationFunction(UnaryOperator<Double> activationFunction) {
        Objects.requireNonNull(activationFunction, "null activation function");
        this.activationFunction = activationFunction;
    }
    
    /**
     * Sets the derivation of activation function for all neurons in hidden layers
     *
     * @param activationFunctionDerivation function to be applied to scalar product of input neurons outputs
     */
    public void setActivationFunctionDerivation(UnaryOperator<Double> activationFunctionDerivation) {
        Objects.requireNonNull(activationFunction, "null activation function");
        this.activationFunctionDerivation = activationFunctionDerivation;
    }
    
    
    /**
     * Sets the method of weight initialization for all connections between neurons
     *
     * @param initialWeigthSupplier function to specify initial weights and biases (before learning)
     */
    public void setInitialWeigthSupplier(DoubleSupplier initialWeigthSupplier) {
        Objects.requireNonNull(initialWeigthSupplier, "null initial weigth supplier");
        this.initialWeigthSupplier = initialWeigthSupplier;
    }

    /**
     * Builds the layered network according to specified parameters (all must be set)
     *
     * @return initialized neural network
     */
    public LayeredNeuralNetwork build() {
        if (architecture == null) {
            throw new IllegalStateException("network architecture not set");
        }
        if (activationFunction == null) {
            throw new IllegalStateException("activation function not set");
        }
        if (initialWeigthSupplier == null) {
            throw new IllegalStateException("initial weight supplier not set");
        }
        InputNeuralNetworkLayer inputLayer = new InputNeuralNetworkLayer(architecture[0]);
        LayeredNeuralNetwork network = new LayeredNeuralNetwork(inputLayer);
        for (int i = 1; i < architecture.length; i++) {
            /*if (i == architecture.length - 1) {
                activationFunction = UnaryOperator.identity();
            }*/
            NeuralNetworkLayer layer = new NeuralNetworkLayer(
                    architecture[i], activationFunction, activationFunctionDerivation, initialWeigthSupplier
            );
            network.addLayer(layer, initialWeigthSupplier);
        }
        return network;
    }
}
