{
	# Fork-only: lets this fork's branch carry reproducible test evidence (the crate's own cargo test suite). Upstream
	# pull requests leave this commit out.
	description = "oxideav-h264 fork with its test suite as a Nix check";
	inputs.nixpkgs.url = "github:NixOS/nixpkgs/335f0738cb2fa9708f3f428e39d2eae975d1338d";
	outputs = { self, nixpkgs }:
		let
			systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
			forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
		in {
			checks = forAll (pkgs: {
				test = pkgs.rustPlatform.buildRustPackage {
					pname = "oxideav-h264-tests";
					version = "0.1.10";
					src = self;
					cargoLock.lockFile = ./Cargo.lock;
					doCheck = true;
					installPhase = "mkdir -p $out";
				};
			});
		};
}
