package cz.muni.fi.pv021.neuralnet;

import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Representation of a formal neuron in an artificial neural network
 *
 * @author David Formanek
 */
public interface Neuron {

    /**
     * Gets access to neurons connected to this as inputs, list can be modified directly
     *
     * @return neural connections ending in this neuron
     */
    List<NeuralConnection> getInputs();

    /**
     * Gets access to neurons connected to this as outputs, list can be modified directly
     *
     * @return neural connections starting in this neuron
     */
    List<NeuralConnection> getOutputs();

    /**
     * Connects this neuron with neuron as parameter (connection object is created automatically)
     *
     * @param neuron output neuron (input is this object)
     * @param weight weight of the connection between this and neuron
     */
    void connectTo(Neuron neuron, double weight);

    /**
     * Checks the input neurons and updates the function value of this neuron
     */
    void computeFunction();

    /**
     * Gets the previously computed value (or constant value for some implementations)
     *
     * @return the function value of this neuron
     */
    double getFunctionOutput();
    
    /**
     * Returns previously computed Delta value. Neuron typically uses delta values of the next-layer
     * neurons (its outputs) to compute its own delta value.
     * @return delta value of the neuron
     */
    public double getDelta();
    
    /**
     * Sets delta value explicitly. This is used for last layer neurons, where delta value is 
     * computed different way
     * @param delta 
     */
    public void computeLastLayerDelta(double delta);
    
    /**
     * Computes delta value of a hidden neuron
     */
    public void computeDelta();
    
    public UnaryOperator getActivationFunctionDerivation();
}
