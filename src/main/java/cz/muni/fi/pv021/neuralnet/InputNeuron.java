package cz.muni.fi.pv021.neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Special neuron that only represents network inputs (data), no own inputs or bias
 *
 * @author David Formanek
 */
public class InputNeuron implements Neuron {

    private final List<NeuralConnection> outputs = new ArrayList<>();
    double value;

    @Override
    public List<NeuralConnection> getInputs() {
        return Collections.emptyList();
    }

    @Override
    public List<NeuralConnection> getOutputs() {
        return outputs;
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
    }

    @Override
    public double getFunctionOutput() {
        return value;
    }

    /**
     * Sets one dimension of input data in the network
     *
     * @param value input data value converted to double in appropriate range
     */
    public void setValue(double value) {
        this.value = value;
    }
}
