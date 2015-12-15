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
    private final UnaryOperator<Double> activationFunctionDerivation;
    private double functionOutput = 0.0;
    private double delta = 0.0;

    /**
     * Constructs neuron with given activation function, ready to be connected with others
     *
     * @param activationFunction function to be applied to scalar product of input neurons outputs
     */
    public BasicNeuron(UnaryOperator<Double> activationFunction
            , UnaryOperator<Double> activationFunctionDerivation) {
        Objects.requireNonNull(activationFunction, "null activation function");
        this.activationFunction = activationFunction;
        this.activationFunctionDerivation = activationFunctionDerivation;
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
        //System.out.println("Suma: " + sum);
        //System.out.println("Suma + actFun: " + this.activationFunction.apply(sum));
        //System.out.println("Suma + tanh: " + Math.tanh(sum));
        this.functionOutput = this.activationFunction.apply(sum);
    }

    @Override
    public double getFunctionOutput() {
        return this.functionOutput;
    }
        @Override
    public double getDelta() {
        return this.delta;
    }
    
    @Override
    public void computeLastLayerDelta(double refOutput) {
        this.delta = -(this.functionOutput - refOutput) * this.activationFunctionDerivation.apply(this.functionOutput);
    }
    
    @Override
    public void computeDelta() {
        
        this.delta = 0;
        for (NeuralConnection connection : this.outputs) {
            this.delta += connection.getOutputNeuron().getDelta() * connection.getWeight();
        }
        this.delta *= this.activationFunctionDerivation.apply(this.functionOutput);
    }
    
    @Override
    public UnaryOperator getActivationFunctionDerivation() {
        return this.activationFunctionDerivation;
    }

}
