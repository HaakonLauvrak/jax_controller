from consys import CONSYS
import matplotlib.pyplot as plt
import config
import jax.numpy as jnp

def main():

    consys = CONSYS()
    error_history, params_history = consys.run_system()
    #plot params history
    kp_history = []
    ki_history = []
    kd_history = []

    for params in params_history:
        kp_history.append(params[0])
        ki_history.append(params[1])
        kd_history.append(params[2])
    
    # Create a figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))  # 2 rows, 1 column

    # Plot error history in the first subplot
    ax1.plot(error_history, label='Error')
    ax1.set_title('Error History')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Error')
    ax1.legend()

    # Plot kp, ki, kd in the second subplot
    ax2.plot(kp_history, label='kp')
    ax2.plot(ki_history, label='ki')
    ax2.plot(kd_history, label='kd')
    ax2.set_title('Parameter Histories')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Parameter Value')
    ax2.legend()

    # Adjust layout
    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(f'./plots/plot_{config.plant}_{config.controller}.png')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()

