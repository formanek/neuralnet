package cz.muni.fi.pv021.neuralnet;

import java.util.Objects;

/**
 * Representation of a connection between two neurons
 *
 * @author David Formanek
 */
public class NeuralConnection {

    private final Neuron input;
    private final Neuron output;
    private double weight;
    private double weightChange;

    /**
     * Connects two neurons
     *
     * @param input input neuron
     * @param output output neuron
     * @param weight the weight of the input neuron for the output neuron
     */
    public NeuralConnection(Neuron input, Neuron output, double weight) {
        Objects.requireNonNull(input, "null input neuron");
        Objects.requireNonNull(output, "null output neuron");
        this.input = input;
        this.output = output;
        this.weight = weight;
        this.weightChange = 0;
    }

    /**
     * Gets the input neuron
     *
     * @return the input neuron
     */
    public Neuron getInputNeuron() {
        return input;
    }

    /**
     * Gets the output neuron
     *
     * @return the output neuron
     */
    public Neuron getOutputNeuron() {
        return output;
    }

    /**
     * Gets the current weight for this connection
     *
     * @return current weight
     */
    public double getWeight() {
        return weight;
    }

    /**
     * Updates the current weight of the connection (useful for learning)
     *
     * @param weight new weight to set
     */
    public void setWeight(double weight) {
        this.weight = weight;
    }
    
    public void clearWeightChange() {
        this.weightChange = 0;
    }
    
    public void incrementWeightChange(double epsilon) {
        /*System.out.println("Inkrement: " + this.output.getDelta() 
                * (double) this.output.getActivationFunctionDerivation().apply(this.output.getFunctionOutput())
                * this.input.getFunctionOutput() + " (" + epsilon + " * " + this.output.getDelta() + " * "
                + this.input.getFunctionOutput() + ")");*/
        this.weightChange += epsilon * this.output.getDelta() 
                * this.input.getFunctionOutput();
    }
    
    public void adaptWeight(double epsilon) {
        this.weight += this.weightChange;
    } 
    
    public double getWeightChange() {
        return this.weightChange;
    }
}
