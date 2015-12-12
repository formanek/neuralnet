package cz.muni.fi.pv021.neuralnet;

import java.util.Random;

/**
 *
 * @author David Formanek
 */
public class Main {

    private static LayeredNeuralNetwork buildNetwork(int... architecture) {
        LayeredNeuralNetworkBuilder builder = new LayeredNeuralNetworkBuilder();
        builder.setArchitecture(architecture);
        builder.setActivationFunction(new HyperbolicTangens(1.7159, 2.0 / 3.0));
        builder.setInitialWeigthSupplier(new UniformRandomInterval(-1.0, 1.0, new Random()));
        return builder.build();
    }
    
    public static void main(String[] args) {
        // usage example
        LayeredNeuralNetwork network = buildNetwork(3, 7, 4);
        network.setData(new double[]{0.25, -0.72, 0.44});
        double[] result = network.getResult();
        for (double s : result) {
            System.out.print(s + " ");
        }
        System.out.println();
        
        // TODO learn, use and evaluate the network
    }
}
