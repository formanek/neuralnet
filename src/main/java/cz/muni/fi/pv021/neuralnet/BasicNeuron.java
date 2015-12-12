package cz.muni.fi.pv021.neuralnet;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Standard neuron for building network (not for input neurons or to represent bias only)
 *
 * @author David Formanek
 */
public class BasicNeuron implements Neuron {

    private final List<NeuralConnection> inputs = new ArrayList<>();
    private final List<NeuralConnection> outputs = new ArrayList<>();
    private final UnaryOperator<Double> activationFunction;
    private double functionOutput = 0.0;

    /**
     * Constructs neuron with given activation function, ready to be connected with others
     *
     * @param activationFunction function to be applied to scalar product of input neurons outputs
     */
    public BasicNeuron(UnaryOperator<Double> activationFunction) {
        Objects.requireNonNull(activationFunction, "null activation function");
        this.activationFunction = activationFunction;
    }

    @Override
    public List<NeuralConnection> getInputs() {
        return inputs;
    }

    @Override
    public List<NeuralConnection> getOutputs() {
        return outputs;
    }

    /**
     * Gets the activation function used in this neuron
     *
     * @return the activation function
     */
    public UnaryOperator<Double> getActivationFunction() {
        return activationFunction;
    }

    @Override
    public void connectTo(Neuron neuron, double weight) {
        Objects.requireNonNull(neuron, "null neuron");
        NeuralConnection connection = new NeuralConnection(this, neuron, weight);
        outputs.add(connection);
        neuron.getInputs().add(connection);
    }

    @Override
    public void computeFunction() {
        double sum = 0.0;
        for (NeuralConnection input : inputs) {
            double value = input.getInputNeuron().getFunctionOutput();
            sum += input.getWeight() * value;
        }
        functionOutput = activationFunction.apply(sum);
    }

    @Override
    public double getFunctionOutput() {
        return functionOutput;
    }
}
