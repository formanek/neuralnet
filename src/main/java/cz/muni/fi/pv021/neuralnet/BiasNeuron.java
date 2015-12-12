package cz.muni.fi.pv021.neuralnet;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Special neuron that allows using bias as if it was weighted output of another neuron
 *
 * @author David Formanek
 */
public class BiasNeuron implements Neuron {

    private final List<NeuralConnection> outputs = new ArrayList<>();

    @Override
    public List<NeuralConnection> getInputs() {
        return Collections.emptyList();
    }

    @Override
    public List<NeuralConnection> getOutputs() {
        return outputs;
    }

    @Override
    public double getFunctionOutput() {
        return 1.0;
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
}
