from diayn_original_cont.networks.vae_regressor import VaeRegressor

from diayn_seq_code_revised.networks.my_gaussian import MyGaussian

class VaeRegressorWithreconvar(VaeRegressor):

    def create_dec(
            self,
            input_dim,
            output_dim,
            hidden_units,
            dropout,
    ) -> MyGaussian:
        """
        Remove default std
        """
        if hidden_units is None:
            dec = MyGaussian(
                input_dim=input_dim,
                output_dim=output_dim,
                dropout=dropout,
            )
        else:
            dec = MyGaussian(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_units=hidden_units,
                dropout=dropout,
            )

        return dec
