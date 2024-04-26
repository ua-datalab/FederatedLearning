# import flwr as fl

# fl.server.start_server(config=fl.server.ServerConfig(num_rounds=1))


import flwr as fl


def aggregate_metrics(metrics):
    """Aggregate metrics by calculating a weighted average."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    aggregated_metrics = {}

    if total_examples > 0:
        for num_examples, client_metrics in metrics:
            for metric_name, metric_value in client_metrics.items():
                if metric_name in aggregated_metrics:
                    aggregated_metrics[metric_name] += metric_value * num_examples
                else:
                    aggregated_metrics[metric_name] = metric_value * num_examples

        for metric_name in aggregated_metrics:
            aggregated_metrics[metric_name] /= total_examples

    return aggregated_metrics


# fl.server.start_server(config=fl.server.ServerConfig(num_rounds=3))


import flwr as fl

def main():
    # Create a FedAvg strategy with a custom metrics aggregation function
    strategy = fl.server.strategy.FedAvg(
        fit_metrics_aggregation_fn=aggregate_metrics
    )

    # Start Flower server with this strategy
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )

if __name__ == "__main__":
    main()
