package cz.muni.fi.pv021.neuralnet;

import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleSupplier;

/**
 * Generator of random double values in a given range (uniform distribution)
 *
 * @author David Formanek
 */
public class UniformRandomInterval implements DoubleSupplier {

    private final double from;
    private final double to;
    private final Random random;

    /**
     * Constructs a generator for uniform distribution in a given range
     *
     * @param from the lower bound of the interval
     * @param to the upper bound of the interval
     * @param random the source of randomness
     */
    public UniformRandomInterval(double from, double to, Random random) {
        if (from > to) {
            throw new IllegalArgumentException("from > to");
        }
        Objects.requireNonNull(random, "randomness source not specified");
        this.from = from;
        this.to = to;
        this.random = random;
    }

    @Override
    public double getAsDouble() {
        return from + (to - from) * random.nextDouble();
    }
}
