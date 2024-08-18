
    template <class T>
    class CyclicBuffer : public std::queue<T>
    {
        int currentIndex = 0;
        int maxNumElements;

    public:

        CyclicBuffer(int newMaxNumElements)
        {
            maxNumElements = newMaxNumElements;
        }

        void addElement (T newElement)
        {
            if (std::queue<T>::size() > maxNumElements)
            {
                std::queue<T>::pop();
            }
            std::queue<T>::push (newElement);
        }
    };

    template <class T>
    class DataPoint
    {
    public:
        std::vector<T> pointDimensionalData;

        DataPoint (std::vector<T> newPointDimensionalData)
        {
            pointDimensionalData = newPointDimensionalData;
        }

        T operator[] (int index)
        {
            return pointDimensionalData [index];
        }
    };

    template <class T>
    class LinearRegressor
    {
    public:

        CyclicBuffer<DataPoint<float>> memory;

        // Let's randomly take 16 samples of memory
        LinearRegressor() : memory (16)
        {

        }

        void updateMemory (DataPoint<float> newElement)
        {
            memory.addElement (newElement);
        }

        float b = 0;
        float a = 0;

        void perform()
        {
            // iterate through all points in memory
            float sumX = 0;
            float sumX2 = 0;
            float sumY = 0;
            float sumXY = 0;

            //This proves that I should have used a vector instead of a queue.
            //Iteration should always be element-wise when it CAN be.
            for (int i = 0; i < memory.size(); ++i)
            {
                DataPoint<float> dataPoint = memory.front();
                sumX += dataPoint[0];
                sumX2 += (dataPoint[0] * dataPoint[0]);
                sumY += (dataPoint[1]);
                sumXY += (dataPoint[0] * dataPoint[1]);
                memory.pop();
                memory.push (dataPoint);
            }

            b = ((float) memory.size() * sumXY - sumX * sumY) / ((float) memory.size() * sumX2 - sumX * sumX);
            a = (sumY - b * sumX) / (float) memory.size();
        }
    };
}
int main()
{
    // Create a LinearRegressor object
    LinearRegressor<float> regressor;

    // Create some sample data points
    std::vector<float> data1 = {1.0, 2.0};
    std::vector<float> data2 = {2.0, 4.0};
    std::vector<float> data3 = {3.0, 6.0};

    // Create DataPoint objects from the sample data
    DataPoint<float> point1(data1);
    DataPoint<float> point2(data2);
    DataPoint<float> point3(data3);

    // Update the memory of the regressor with the data points
    regressor.updateMemory(point1);
    regressor.updateMemory(point2);
    regressor.updateMemory(point3);

    // Perform the regression
    regressor.perform();

    // Print the calculated coefficients
    std::cout << "a: " << regressor.a << std::endl;
    std::cout << "b: " << regressor.b << std::endl;

    return 0;
}