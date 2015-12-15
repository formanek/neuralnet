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

    private static LayeredNeuralNetwork buildNetwork(int... architecture) {
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
    
    private static void classificationRSA() throws IOException{
        List<DatasetExample> dataset = InputDataCSV.getDataFromFile("modulus_learning_set_processed.txt");
        /*for (DatasetExample example : dataset) {
            System.out.println(example.toString() + "\n");
        }*/
        LayeredNeuralNetwork network = buildNetwork(11, 15, 15, 7);
        Trainer trainer = new Trainer(network, dataset, (double) 0.9);
        trainer.train(40);
    }

    private static void classificationTrivial() throws IOException{
        List<DatasetExample> dataset = InputDataCSV.getDataFromFile("trivial.txt");
        /*for (DatasetExample example : dataset) {
            System.out.println(example.toString() + "\n");
        }*/
        LayeredNeuralNetwork network = buildNetwork(6, 8, 1);
        Trainer trainer = new Trainer(network, dataset, (double) 0.5);
        trainer.train(1000);
    }
    
    public static void main(String[] args) throws IOException {
        try {
            PrintStream out = new PrintStream(new FileOutputStream("netBeansOutput.txt"));
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
        
        
        
        
        
        
        //Main.classificationRSA();
        Main.classificationTrivial();
        // TODO learn, use and evaluate the network
    }

}
