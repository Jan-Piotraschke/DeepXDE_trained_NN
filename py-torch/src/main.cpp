#include <torch/script.h>
#include <iostream>
#include <vector>
#include <fstream>


int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // Define the number of steps
    const int num_steps = 40000;
    // Create a vector to store the range
    std::vector<float> data(num_steps);
    // populate the vector with data ranging from 0 to 40 using 0.001 stepsize
    for (int i = 0; i < num_steps; ++i) {
        data[i] = i * 0.001f;
    }

    // Convert std::vector to a single torch::Tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor input_tensor = torch::from_blob(data.data(), {num_steps, 1}, options);

    // Execute the model and turn its output into a tensor.
    at::Tensor output = module.forward({input_tensor}).toTensor();
    // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    // Save the output to a csv file
    std::ofstream outfile("output.csv");
    if (!outfile.is_open()) {
        std::cerr << "Failed to create the file\n";
        return -1;
    }

    // Get the size of the output tensor assuming it is 2D
    auto sizes = output.sizes();
    int64_t num_rows = sizes[0];
    int64_t num_cols = sizes[1];

    // Iterate over the tensor and write the contents to the file.
    for (int64_t i = 0; i < num_rows; ++i) {
        for (int64_t j = 0; j < num_cols; ++j) {
            // Write the element and a comma unless it's the last element in the row
            outfile << output[i][j].item<float>();
            if (j != num_cols - 1) {
                outfile << ",";
            }
        }
        // End of the row, write a new line
        outfile << "\n";
    }

    // Close the file
    outfile.close();

    std::cout << "ok\n";

    return 0;
}
