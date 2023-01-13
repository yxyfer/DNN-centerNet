#include <torch/torch.h>

std::vector<torch::Tensor> top_pool_forward(
    torch::Tensor input) {
    // Initialize output
    torch::Tensor output = torch::zeros_like(input);

    // Get height
    int64_t height = input.size(2);

    // Copy the last column
    torch::Tensor input_temp  = input.slice(2, height - 1);
    torch::Tensor output_temp = output.slice(2, height - 1);
    output_temp.copy_(input_temp);

    torch::Tensor max_temp;
    for (int64_t ind = 1; ind < height; ++ind) {
        input_temp  = input.slice(2, height - ind - 1);
        output_temp = output.slice(2, height - ind);
        max_temp    = output.slice(2, height - ind - 1);

        torch::max_out(max_temp, input_temp, output_temp);
    }
    
    return { output };
}


std::vector<torch::Tensor> top_pool_backward(
    torch::Tensor input,
    torch::Tensor grad_output) {
    auto output = torch::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t height  = input.size(2);
    int32_t width   = input.size(3);

    auto max_val = torch::zeros({batch, channel, width}, torch::kFloat32);
    auto max_ind = torch::zeros({batch, channel, width}, torch::kInt64);

    auto input_temp = input.slice(2, height - 1);
    max_val.copy_(input_temp);

    max_ind.fill_(height - 1);

    auto output_temp = output.slice(2, height - 1);
    auto grad_output_temp = grad_output.slice(2, height - 1);
    output_temp.copy_(grad_output_temp);

    auto un_max_ind = max_ind.unsqueeze(2);
    auto gt_mask    = torch::zeros({batch, channel, width}, torch::kByte);
    auto max_temp   = torch::zeros({batch, channel, width}, torch::kFloat32);

    for (int32_t ind = 1; ind < height; ++ind) {
        input_temp  = input.slice(2, height - ind - 1);
        torch::gt_out(gt_mask, input_temp, max_val);

        torch::masked_select_out(max_temp, input_temp, gt_mask);
        max_val.masked_scatter_(gt_mask, max_temp);
        max_ind.masked_fill_(gt_mask, height - ind - 1);

        grad_output_temp = grad_output.slice(2, height - ind - 1);
        output.scatter_add_(2, un_max_ind, grad_output_temp.unsqueeze(2));
    }

    return { output };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &top_pool_forward, "Top Pool Forward",
        py::call_guard<py::gil_scoped_release>()    
    );
    m.def(
        "backward", &top_pool_backward, "Top Pool Backward",
        py::call_guard<py::gil_scoped_release>()
    );
}
