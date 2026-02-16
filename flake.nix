{
  description = "Flake for openavmkit";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        pyPkgs = pkgs.python311Packages;
        version = if self ? shortRev then "0.0.0+" + self.shortRev else "0.0.0";

        openavmkit = pyPkgs.buildPythonPackage {
          pname = "openavmkit";
          inherit version;
          src = ./.;
          format = "setuptools";

          nativeBuildInputs = with pyPkgs; [
            setuptools-scm
          ];

          # Ensure deterministic versioning in Nix builds when .git metadata is absent.
          SETUPTOOLS_SCM_PRETEND_VERSION = version;

          pythonImportsCheck = [ "openavmkit" ];
          doCheck = false;
        };
      in {
        packages.default = openavmkit;
        packages.openavmkit = openavmkit;

        devShells.default = pkgs.mkShell {
          inputsFrom = [ openavmkit ];
          packages = with pkgs; [
            python311
            git
            pre-commit
          ];

          shellHook = ''
            echo "Welcome to the openavmkit dev shell!"
            export PYTHONPATH=$PWD
          '';
        };
      });
}
