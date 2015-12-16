package cz.muni.fi.pv021.neuralnet;

import cz.muni.fi.pv021.neuralnet.inputData.InputDataCSV;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Random;
import java.util.function.UnaryOperator;

/**
 *
 * @author David Formanek
 */
public class Main {
    
    private static Configuration conf;

    private static LayeredNeuralNetwork buildNetworkTanh(double minWeight, double maxWeight, int... architecture) {
        LayeredNeuralNetworkBuilder builder = new LayeredNeuralNetworkBuilder();
        builder.setArchitecture(architecture);
        builder.setActivationFunction(new HyperbolicTangens(1.7159, 2.0 / 3.0));
        builder.setActivationFunctionDerivation(new HyperbolicTangensDerivation(1.7159, 2.0 / 3.0));
        
        builder.setInitialWeigthSupplier(new UniformRandomInterval(-1.0, 1.0, new Random()));
        //builder.setInitialWeigthSupplier(new UniformRandomInterval(-0.0, 0.0, new Random())); // pro porovnani s Python verzi
        
        return builder.build();
    }
    
    private static LayeredNeuralNetwork buildNetworkSigmoid(double minWeight, double maxWeight, int... architecture) {
        LayeredNeuralNetworkBuilder builder = new LayeredNeuralNetworkBuilder();
        builder.setArchitecture(architecture);
        //builder.setActivationFunction(new HyperbolicTangens(1.7159, 2.0 / 3.0));
        //builder.setActivationFunctionDerivation(new HyperbolicTangensDerivation(1.7159, 2.0 / 3.0));
        builder.setActivationFunction(new Sigmoid(1.0));
        builder.setActivationFunctionDerivation(new SigmoidDerivation(1.0));
        
        builder.setInitialWeigthSupplier(new UniformRandomInterval(0.0, 1.0, new Random()));
        //builder.setInitialWeigthSupplier(new UniformRandomInterval(-0.0, 0.0, new Random())); // pro porovnani s Python verzi
        
        return builder.build();
    }
    
    public static void main(String[] args) throws IOException {
        
        //toto presmeruje vysstup do souboru
        try {
            PrintStream out = new PrintStream(new FileOutputStream("outputFile.txt"));
            System.setOut(out);
        } catch (Exception e) {
            System.out.println(e);
        }
        
        
        // usage example
        /*LayeredNeuralNetwork network = buildNetwork(3, 7, 4);
        network.setData(new double[]{0.25, -0.72, 0.44});
        double[] result = network.getResult();
        for (double s : result) {
            System.out.print(s + " ");
        }
        System.out.println();
        */
        
        
        
        if (args.length == 1) {
            try {
                conf = new Configuration(args[0]);
                //System.out.println("Settings loaded: " + conf.toString());
                if(conf.getActivationFunction().equalsIgnoreCase("sigmoid")) {
                    LayeredNeuralNetwork network = buildNetworkSigmoid(conf.getMinWeightInit(), conf.getMaxWeightInit(), conf.getArchitecture());
                    Trainer trainer = new Trainer(network, conf);
                    trainer.train();
                    trainer.test();
                } else if (conf.getActivationFunction().equalsIgnoreCase("tanh")) {
                    LayeredNeuralNetwork network = buildNetworkTanh(conf.getMinWeightInit(), conf.getMaxWeightInit(), conf.getArchitecture());
                    Trainer trainer = new Trainer(network, conf);
                    trainer.train();
                    trainer.test();
                }

            } catch (Exception e) {
                System.err.println("Error while loading settings:\n\t" + e);
            }
            
        } else {
            System.out.println("HELP:\nPlease pass the configuration file as the only parameter.");
        }

    }

}
