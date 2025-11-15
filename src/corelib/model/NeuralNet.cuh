#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <curand_kernel.h>
#include "../interfaces/ITraits.hpp"
#include "../AlignedVec.hpp"
#include <string>
#include <math.h>
#include <fstream>
#include <iostream>
#include <time.h>
#include <numeric>

/*
// Initialiser les poids et biais
__global__ void initRand(unsigned long long seed, int width, int height, float* mat)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < height && column < width)
	{
		int idx = row * width + column;
		curandState state;
		curand_init(seed + idx, 0, 0, &state);  // Utilisation d'un décalage dans le seed

		// Génération d'un nombre aléatoire avec une distribution normale
		mat[idx] = curand_normal(&state) * sqrtf(2.0f / (width * height));
	}
}
*/

template<typename GameTag>
struct StateActionT
{
	ObsStateT<GameTag> state;
	ActionT<GameTag>   lastAction;
};
template<typename GameTag>
struct IdxStateActionT
{
	IdxStateT<GameTag>  state;
	IdxActionT<GameTag> lastAction;
};
template<typename GameTag>
struct ModelResults
{
	AlignedVec<float> values;		// values for every players at the leaf
	AlignedVec<float> policy;		// policy at the leaf
	AlignedVec<float> belief;		// buffer for belief probabilities

	ModelResults()
		: values(ITraits<GameTag>::kNumPlayers)
		, policy(ITraits<GameTag>::kActionSpace)
		, belief(ITraits<GameTag>::kNumElems * ITraits<GameTag>::kNumPos)
	{
	}
};

template<typename GameTag>
class NeuralNet
{
private:
	using GT = ITraits<GameTag>;
	using ObsState = typename ObsStateT<GameTag>;
	using Action = typename ActionT<GameTag>;
	using IdxState = typename IdxStateT<GameTag>;
	using IdxAction = typename IdxActionT<GameTag>;
	using StateAction = typename StateActionT<GameTag>;
	using IdxStateAction = typename IdxStateActionT<GameTag>;

	const uint16_t m_batchSize;
	const uint16_t m_historySize = 8;

	/*
	const uint64_t seed;
	const uint32_t inputSize;								// Taille de l'entrée
	const uint32_t outputSize;								// Taille de la sortie

	const uint32_t hiddenSize;								// Nombre de neurone dans les couches cachées
	const uint32_t nbHiddenLayers;							// Nombre de couche cachée

	const float momentumNorm;
	const float gammaNorm;
	const float betaNorm;
	const float dropoutRate;

	const float huberLossDelta = 1.f;
	const float lambdaL1;
	const float lambdaL2;
	const float weightDecay;
	const float gradientClip;

	dim3 dimBlock;									// Taille des blocs CUDA
	const uint32_t batchSize;

	int t = 1;

	// Pointeurs des Poids, biais et activations de chaque couche sur le GPU
	Tensor<uint32_t> layerSizes;       // +1 pour la couche d'entrée

	float* d_weights;
	float* d_biases;

	float* d_activations;
	float* d_errors;
	float* d_maskDropout;

	float* d_mWeights;
	float* d_vWeights;
	float* d_mBiases;
	float* d_vBiases;

	float* d_targets;
	float* d_regPenalties;
	float* d_losses;

	Tensor<float> h_output;			// Output de chaque sortie du batch
	Tensor<float> h_losses;			// Pertes de chaque sortie du batch

	Tensor<uint32_t> matOffsets;
	Tensor<uint32_t> vectOffsets;

	void displayMatGPU(int size, const float* mat) const
	{
		float* temp = new float[size];
		cudaMemcpy(temp, mat, size * sizeof(float), cudaMemcpyDeviceToHost);

		for (int i = 0; i < size; i++) std::cout << temp[i] << " ";
		delete[] temp;
	}

	void displayMatCPU(int size, const float* mat) const
	{
		for (int i = 0; i < size; i++) std::cout << mat[i] << " ";
	}
	*/

public:
	NeuralNet(uint16_t batchSize) : m_batchSize(batchSize)
	{
	}

	void forwardBatch(const AlignedVec<IdxStateAction>& inferenceBuf,
		AlignedVec<ModelResults<GameTag>>& resultsBuf)
	{
		//std::cout << "forwardBatch called with " << resultsBuf.size() << " inferences" << std::endl;

		// Le nombre d'inférences à faire correspond au nombre d'historiques
		// Chaque historique représente un leaf node à évaluer
		const size_t numInferences = resultsBuf.size();

		// Vérifier que nous avons assez de données d'entrée
		// inferenceBuf contient l'historique de chaque inference (historySize par inference)

		// Pour chaque inference dans le batch
		for (size_t b = 0; b < numInferences; ++b)
		{
			ModelResults<GameTag>& results = resultsBuf[b];

			// === POLICY (distribution de probabilité sur les actions) ===
			// Générer des logits aléatoires
			for (int i = 0; i < GT::kActionSpace; ++i)
			{
				results.policy[i] = static_cast<float>(rand()) / RAND_MAX;
			}

			// Softmax pour normaliser en probabilités
			float maxLogit = *std::max_element(results.policy.begin(), results.policy.end());
			float sum = 0.0f;

			for (int i = 0; i < GT::kActionSpace; ++i)
			{
				results.policy[i] = std::exp(results.policy[i] - maxLogit);
				sum += results.policy[i];
			}

			for (int i = 0; i < GT::kActionSpace; ++i)
			{
				results.policy[i] /= sum;
			}

			// === VALUES (valeurs pour chaque joueur) ===
			for (uint8_t p = 0; p < GT::kNumPlayers; ++p)
			{
				// Valeurs entre -1 et 1 (typique pour les jeux à somme nulle)
				results.values[p] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
			}

			// === BELIEF (probabilités pour les éléments cachés) ===
			// Si tu utilises le belief, sinon on peut ignorer pour l'instant
			if (results.belief.size() > 0)
			{
				float beliefSum = 0.0f;
				for (size_t i = 0; i < results.belief.size(); ++i)
				{
					results.belief[i] = static_cast<float>(rand()) / RAND_MAX;
					beliefSum += results.belief[i];
				}

				// Normaliser en probabilités
				if (beliefSum > 0.0f)
				{
					for (size_t i = 0; i < results.belief.size(); ++i)
					{
						results.belief[i] /= beliefSum;
					}
				}
			}
		}
	}

	// Version alternative plus simple si tu veux juste tester
	void forwardBatchSimple(const AlignedVec<IdxStateAction>& inferenceBuf,
		AlignedVec<ModelResults<GameTag>>& resultsBuf)
	{
		const size_t numInferences = resultsBuf.size();

		for (size_t b = 0; b < numInferences; ++b)
		{
			ModelResults<GameTag>& results = resultsBuf[b];

			// Policy uniforme
			float uniformProb = 1.0f / static_cast<float>(GT::kActionSpace);
			for (int i = 0; i < GT::kActionSpace; ++i)
			{
				results.policy[i] = uniformProb;
			}

			// Values aléatoires entre -1 et 1
			for (uint8_t p = 0; p < GT::kNumPlayers; ++p)
			{
				results.values[p] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
			}

			// Belief uniforme si nécessaire
			if (results.belief.size() > 0)
			{
				float uniformBelief = 1.0f / static_cast<float>(results.belief.size());
				std::fill(results.belief.begin(), results.belief.end(), uniformBelief);
			}
		}
	}

	void normalizeToProba(const AlignedVec<IdxAction>& validIdxActions, float* policy)
	{

	}

	/*
	{
		// C++ pseudo
		size_t total_needed = compute_total_bytes(...);
		void* device_pool = nullptr;
		cudaMalloc(&device_pool, total_needed);

		// create offsets
		char* base = static_cast<char*>(device_pool);
		char* ptr = base;

		// params
		W_proj_dev = reinterpret_cast<float16_t*>(ptr); ptr += bytes_W_proj;
		fact_embeddings_dev = reinterpret_cast<float16_t*>(ptr); ptr += bytes_fact_embeddings;
		// ... per-layer params
		// activations/scratch
		scratch_dev = reinterpret_cast<uint8_t*>(ptr); ptr += bytes_scratch;
		// outputs
		belief_logits_dev = reinterpret_cast<float*>(ptr); ptr += bytes_belief;
		policy_logits_dev = reinterpret_cast<float*>(ptr); ptr += bytes_policy;




		// includes cuda_runtime.h, cublas_v2.h, etc.
		cudaStream_t stream;
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

		// 1) allocate big device pool
		size_t total_bytes = compute_total_bytes(...);
		void* device_pool;
		cudaMalloc(&device_pool, total_bytes);

		// 2) carve offsets (example)
		char* base = static_cast<char*>(device_pool);
		auto align_up = [](size_t n, size_t align) { return (n + align - 1) & ~(align - 1); };
		size_t offset = 0;
		size_t align = 256;
		offset = align_up(offset, align); W_proj_dev = (float16_t*)(base + offset); offset += bytes_W_proj;
		offset = align_up(offset, align); fact_embeddings_dev = (float16_t*)(base + offset); offset += bytes_fact_embeddings;
		offset = align_up(offset, align); scratch_dev = (uint8_t*)(base + offset); offset += bytes_scratch;
		offset = align_up(offset, align); belief_logits_dev = (float*)(base + offset); offset += bytes_belief;
		// ...

		// 3) allocate host pinned inputs/outputs
		void* host_in_idx;
		cudaHostAlloc(&host_in_idx, size_host_in_idx, cudaHostAllocDefault);
		void* host_out_belief;
		cudaHostAlloc(&host_out_belief, size_host_out_belief, cudaHostAllocDefault);

		// fill host_in_idx with batch data...

		// 4) copy indices to GPU async
		cudaMemcpyAsync(device_pool + offset_idx, host_in_idx, size_host_in_idx, cudaMemcpyHostToDevice, stream);

		// 5) launch gather kernel (device: read embeddings table & indices -> write emb_fact_vectors)
		gatherEmbeddingsKernel << <grid, block, 0, stream >> > (  pointers: embeddings_dev, indices_dev, out_emb_dev, B*S, ... );

		// 6) run GEMMs / transformer on stream (cublas handles with stream)
		cublasSetStream(handle, stream);
		cublasGemmStridedBatched(...);

		// 7) after final heads, copy outputs back async
		cudaMemcpyAsync(host_out_belief, belief_logits_dev, size_host_out_belief, cudaMemcpyDeviceToHost, stream);

		// 8) synchronize (or use event)
		cudaStreamSynchronize(stream);

		// host_out_belief is ready to use on CPU

	}
	*/
};

/*
void Network::initializeWeightsAndBiases() const {
	// Initialisation des poids pour chaque couche
	for (int layer = 0; layer < nbHiddenLayers + 1; layer++) {
		int inSize = layerSizes[layer];       // Taille de la couche précédente (entrée)
		int outSize = layerSizes[layer + 1];  // Taille de la couche actuelle (sortie)

		// Calculer la taille de la grille en utilisant les bonnes dimensions
		dim3 dimGrid((outSize - 1) / dimBlock.x + 1,
			(inSize - 1) / dimBlock.y + 1, 1);

		// Initialisation aléatoire des poids avec distribution normale
		initRand << <dimGrid, dimBlock >> > (seed, outSize, inSize, d_weights + matOffsets[layer]);

		// Initialisation des biais et des moyennes au GPU
		cudaMemset(d_biases + vectOffsets[layer], 0, outSize * sizeof(float));
		cudaMemset(d_mWeights + matOffsets[layer], 0, outSize * inSize * sizeof(float));
		cudaMemset(d_vWeights + matOffsets[layer], 0, outSize * inSize * sizeof(float));
		cudaMemset(d_mBiases + vectOffsets[layer], 0, outSize * sizeof(float));
		cudaMemset(d_vBiases + vectOffsets[layer], 0, outSize * sizeof(float));
	}

	// Synchronisation pour garantir la fin des opérations CUDA
	cudaDeviceSynchronize();
}

////////// Forward //////////

__global__ void forward(int n, int out_w,
	const float* input, const float* weights, const float* biases, float* output) {
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (column < out_w) {
		float out = biases[column];
		for (int i = 0; i < n; i++) {
			out += weights[i * out_w + column] * input[i];
		}
		output[column] = out;  // Stocke la sortie pour chaque neurone
	}
}

__global__ void forwardRelu(int inSize, int outSize,
	const float* input, const float* weights, const float* biases, float* output) {
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (column < outSize) {
		float out = biases[column];
		for (int i = 0; i < inSize; i++) {
			out += weights[i * outSize + column] * input[i];
		}
		// ReLU activation
		output[column] = out > 0.f ? out : 0.f;
	}
}

__global__ void softmax(int size, float* in, float* out) {
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (col < size) {
		// Trouve la valeur maximale pour la stabilité numérique
		float maxval = in[0];
		for (int i = 1; i < size; i++) {
			maxval = fmaxf(maxval, in[i]);
		}

		// Calcul du diviseur
		float divisor = 0.f;
		for (int i = 0; i < size; i++) {
			divisor += expf(in[i] - maxval);
		}

		// Applique la fonction softmax pour chaque sortie
		out[col] = expf(in[col] - maxval) / divisor;
	}
}

float* Network::forwardNetwork(const float* input) {
	// Copier le batch d'entrées (inputsBatch) sur le GPU
	cudaMemcpy(d_activations, input, layerSizes[0] * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// Boucle pour passer par chaque couche cachée
	for (int layer = 0; layer < nbHiddenLayers; layer++) {
		// Calcul des dimensions pour CUDA (nombre de blocs et de threads)
		dim3 dimGrid((layerSizes[layer + 1] - 1) / dimBlock.x + 1, 1, 1);

		// Lancer le kernel pour la couche cachée avec ReLU comme fonction d'activation
		forwardRelu << <dimGrid, dimBlock >> > (
			layerSizes[layer],
			layerSizes[layer + 1],
			d_activations + vectOffsets[layer],
			d_weights + matOffsets[layer],
			d_biases + vectOffsets[layer],
			d_activations + vectOffsets[layer + 1]);
		cudaDeviceSynchronize();
	}

	// Calcul pour la couche de sortie
	dim3 dimGrid((layerSizes[nbHiddenLayers + 1] - 1) / dimBlock.x + 1, 1, 1);

	// Lancer le kernel pour la couche de sortie (sans ReLU)
	forward << <dimGrid, dimBlock >> > (
		layerSizes[nbHiddenLayers],							// Nombre d'entrées de la couche précédente
		layerSizes[nbHiddenLayers + 1],						// Nombre de sorties de la couche actuelle
		d_activations + vectOffsets[nbHiddenLayers],        // Pointeur d'entrée
		d_weights + matOffsets[nbHiddenLayers],				// Pointeur des poids
		d_biases + vectOffsets[nbHiddenLayers],             // Pointeur des biais
		d_activations + vectOffsets[nbHiddenLayers + 1]);	// Pointeur de sortie
	cudaDeviceSynchronize();

	// Lancer le kernel pour appliquer la fonction softmax sur la sortie
	softmax << <dimGrid, dimBlock >> > (
		layerSizes[nbHiddenLayers + 1],
		d_activations + vectOffsets[nbHiddenLayers + 1],
		d_activations + vectOffsets[nbHiddenLayers + 1]);
	cudaDeviceSynchronize();

	cudaMemcpy(
		h_output,
		d_activations + vectOffsets[nbHiddenLayers + 1],
		layerSizes[nbHiddenLayers + 1] * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Retourner la sortie softmax depuis la mémoire hôte
	return h_output;
}

////////// Forward pour un Batch //////////

__global__ void forwardBatch(int batchSize, int inSize, int outSize,
	const float* input, const float* weights, const float* biases, float* output) {

	int batchIdx = blockIdx.y * blockDim.y + threadIdx.y; // Indice du batch
	int column = blockIdx.x * blockDim.x + threadIdx.x; // Indice de la sortie

	// Vérifie que l'indice de batch et de sortie sont dans les limites
	if (batchIdx < batchSize && column < outSize) {
		// Initialise la sortie avec le biais
		float out = biases[column];

		// Accumule la somme pondérée de chaque entrée
		for (int i = 0; i < inSize; i++) {
			out += weights[i * outSize + column] * input[batchIdx * inSize + i];
		}

		// Applique l'activation ReLU
		output[batchIdx * outSize + column] = out;
	}
}

__global__ void forwardReluBatch(int batchSize, int inSize, int outSize, float dropoutRate,
	const float* input, const float* weights, const float* biases, float* output, float* maskDropout) {

	int batchIdx = blockIdx.y * blockDim.y + threadIdx.y; // Indice du batch
	int column = blockIdx.x * blockDim.x + threadIdx.x; // Indice de la sortie

	// Vérifie que l'indice de batch et de sortie sont dans les limites
	if (batchIdx < batchSize && column < outSize) {
		// Initialise la sortie avec le biais
		float out = biases[column];

		// Accumule la somme pondérée de chaque entrée
		for (int i = 0; i < inSize; i++) {
			out += weights[i * outSize + column] * input[batchIdx * inSize + i];
		}

		// Applique l'activation ReLU et le mask du dropout
		output[batchIdx * outSize + column] = out > 0.0f ? out * maskDropout[batchIdx * outSize + column] / (1.0f - dropoutRate) : 0.0f;
	}
}

__global__ void setMaskDropout(unsigned long long seed, int batchSize, int size, float dropoutRate, float* mask)
{
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < batchSize && column < size)
	{
		int idx = row * size + column;
		curandState state;
		curand_init(seed + idx, 0, 0, &state);  // Utilisation d'un décalage dans le seed

		// Génération d'un nombre aléatoire avec une distribution normale
		mask[idx] = curand_normal(&state) > dropoutRate ? 1.0f : 0.0f;
	}
}

__global__ void softmaxBatch(int batchSize, int size, float* in, float* out)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	if (row < batchSize && col < size)
	{
		float maxval = in[row * size];
		for (int i = 1; i < size; i++)
		{
			maxval = fmaxf(maxval, in[row * size + i]);
		}
		float divisor = 0.f;
		for (int i = 0; i < size; i++)
		{
			divisor += expf(in[row * size + i] - maxval);
		}
		out[row * size + col] = expf(in[row * size + col] - maxval) / (divisor);
	}
}

float* Network::forwardBatchNetwork(const float* inputsBatch) {
	// Copier chaque entrée du batch individuellement dans d_activations[0]
	cudaMemcpy(d_activations, inputsBatch, inputSize * batchSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	// Boucle pour passer par chaque couche cachée
	for (int layer = 0; layer < nbHiddenLayers; layer++) {
		// Calcul des dimensions pour CUDA (nombre de blocs et de threads)
		dim3 dimGrid((layerSizes[layer + 1] - 1) / dimBlock.x + 1, (batchSize - 1) / dimBlock.y + 1, 1);

		// Lancer le kernel pour le mask du dropout
		setMaskDropout << <dimGrid, dimBlock >> > (seed,
			batchSize,
			layerSizes[layer + 1],
			dropoutRate,
			d_maskDropout + vectOffsets[layer + 1]);

		// Lancer le kernel pour la couche cachée avec ReLU comme fonction d'activation
		forwardReluBatch << <dimGrid, dimBlock >> > (
			batchSize,
			layerSizes[layer],
			layerSizes[layer + 1],
			dropoutRate,
			d_activations + vectOffsets[layer],
			d_weights + matOffsets[layer],
			d_biases + vectOffsets[layer],
			d_activations + vectOffsets[layer + 1],
			d_maskDropout + vectOffsets[layer + 1]);

		cudaDeviceSynchronize();
	}

	// Calcul pour la couche de sortie
	dim3 dimGrid((layerSizes[nbHiddenLayers + 1] - 1) / dimBlock.x + 1, (batchSize - 1) / dimBlock.y + 1, 1);

	// Lancer le kernel pour la couche de sortie (sans ReLU)
	forwardBatch << <dimGrid, dimBlock >> > (
		batchSize,
		layerSizes[nbHiddenLayers],
		layerSizes[nbHiddenLayers + 1],
		d_activations + vectOffsets[nbHiddenLayers],
		d_weights + matOffsets[nbHiddenLayers],
		d_biases + vectOffsets[nbHiddenLayers],
		d_activations + vectOffsets[nbHiddenLayers + 1]);
	cudaDeviceSynchronize();

	// Lancer le kernel pour appliquer la fonction softmax sur la sortie
	softmaxBatch << <dimGrid, dimBlock >> > (
		batchSize,
		layerSizes[nbHiddenLayers + 1],
		d_activations + vectOffsets[nbHiddenLayers],
		d_activations + vectOffsets[nbHiddenLayers + 1]);
	cudaDeviceSynchronize();

	
	const int sizeTemp = outputSize * batchSize;
	float* temp = new float[sizeTemp];
	cudaMemcpy(temp, d_activations[nbHiddenLayers + 1], sizeTemp * sizeof(float), cudaMemcpyDeviceToHost);
	displayMatrix(1, sizeTemp, temp);
	delete[] temp;

	// Copie du résultat sur Host
	//cudaMemcpy(h_output, d_activations[nbHiddenLayers + 1], outputSize * batchSize * sizeof(float), cudaMemcpyDeviceToHost);

	cudaMemcpy(
		h_output,
		d_activations + vectOffsets[nbHiddenLayers + 1],
		outputSize * batchSize * sizeof(float),
		cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	// Retourner la sortie softmax depuis la mémoire hôte
	return h_output;
}

////////// Backward pour un Batch //////////

__global__ void regL1L2(int width, int height, float* d_weights, float* d_regSum, float lambdaL1, float lambdaL2) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = blockIdx.y * width * height;  // Correction pour multi-couches

	if (idx < width * height) {
		float weight = d_weights[offset + idx];
		atomicAdd(d_regSum, lambdaL1 * fabsf(weight) + lambdaL2 * weight * weight);
	}
}

__global__ void huberLossBatch(int batchSize, int outSize,
	const float* preds, const float* real, float* losses, float* errors,
	float delta, float regPenalty) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int batchIdx = idx / outSize;

	if (idx < outSize * batchSize) {
		// Protection valeur numérique
		float pred = fmaxf(fminf(preds[idx], 1e30f), -1e30f);
		float target = fmaxf(fminf(real[idx], 1e30f), -1e30f);

		float err = pred - target;
		errors[idx] = err;

		float absErr = fabsf(err);
		float loss = (absErr <= delta) ? 0.5f * err * err : delta * (absErr - 0.5f * delta);

		// Régularisation normalisée par élément
		loss += regPenalty / (outSize * batchSize);

		atomicAdd(&losses[batchIdx], loss);
	}
}

__global__ void backwardBatch(int layer, int batchSize, int inSize, int outSize,
	float gradientClip, float dropoutRate, float* weights, float* d_error,
	float* out_d_error, float* activations, float* maskDropout)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx < outSize && idy < batchSize) {
		float dl = 0.0f;

		// Boucle sécurisée avec limite explicite
		for (int i = 0; i < inSize; i++) { // layerSizes à passer en paramètre constant
			int weight_idx = i * outSize + idx;
			if (weight_idx < inSize * outSize) {
				dl += weights[weight_idx] * d_error[idy * inSize + i];
			}
		}

		// Clipping agressif avec vérification
		dl = isnan(dl) ? 0.0f : fmaxf(fminf(dl, gradientClip), -gradientClip);

		// Dérivée ReLU stabilisée
		float activation = activations[idy * outSize + idx];
		float derivative = (activation > 1e-7f) ? 1.0f : 0.0f;

		// Dropout sécurisé
		float mask = (layer > 0 && dropoutRate > 0.0f && dropoutRate < 0.99f) ?
			maskDropout[idy * outSize + idx] / (1.0f - dropoutRate) : 1.0f;

		out_d_error[idy * outSize + idx] = dl * derivative * mask;
	}
}

float Network::backwardBatchNetwork(const float* targets) {
	// 1. Réinitialisation protégée
	cudaMemset(d_losses, 0, batchSize * sizeof(float));
	cudaMemset(d_regPenalties, 0, sizeof(float));
	cudaDeviceSynchronize();

	// 2. Calcul régularisation couche par couche
	for (int layer = 0; layer < nbHiddenLayers + 1; layer++) {
		dim3 dimGrid(
			(layerSizes[layer + 1] * layerSizes[layer] + dimBlock.x - 1) / dimBlock.x,
			1  // Une grille 1D pour cette version
		);

		regL1L2 << <dimGrid, dimBlock >> > (
			layerSizes[layer + 1],
			layerSizes[layer],
			d_weights + matOffsets[layer],
			d_regPenalties + layer,  // Stockage séparé par couche
			lambdaL1,
			lambdaL2
			);
		cudaDeviceSynchronize();
	}

	// 3. Somme finale des régularisations
	float regTotal = 0.0f;
	float* h_regPenalties = new float[nbHiddenLayers + 1];
	cudaMemcpy(h_regPenalties, d_regPenalties, (nbHiddenLayers + 1) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	for (int i = 0; i <= nbHiddenLayers; i++) regTotal += h_regPenalties[i];
	delete[] h_regPenalties;

	// 4. Calcul perte avec paramètres sécurisés
	cudaMemcpy(d_targets, targets, outputSize * batchSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	//displayMatGPU(outputSize * batchSize, d_targets);

	dim3 dimGridLoss((outputSize * batchSize + dimBlock.x - 1) / dimBlock.x, 1);
	huberLossBatch << <dimGridLoss, dimBlock >> > (
		batchSize,
		outputSize,
		d_activations + vectOffsets[nbHiddenLayers + 1],
		d_targets,
		d_losses,
		d_errors + vectOffsets[nbHiddenLayers + 1],
		huberLossDelta,
		regTotal  // Already normalized in kernel
		);
	cudaDeviceSynchronize();

	// 5. Rétropropagation dimensionnée précisément
	for (int layer = nbHiddenLayers; layer >= 0; --layer) {
		dim3 dimGridBp(
			(layerSizes[layer + 1] + dimBlock.x - 1) / dimBlock.x,
			(batchSize + dimBlock.y - 1) / dimBlock.y
		);

		backwardBatch << <dimGridBp, dimBlock >> > (
			layer,
			batchSize,
			layerSizes[layer],
			layerSizes[layer + 1],
			gradientClip,
			dropoutRate,
			d_weights + matOffsets[layer],
			d_errors + vectOffsets[layer + 1],
			d_errors + vectOffsets[layer],
			d_activations + vectOffsets[layer + 1],
			d_maskDropout + vectOffsets[layer]
			);
		cudaDeviceSynchronize();
	}

	// 6. Vérification finale robuste
	cudaMemcpy(h_losses, d_losses, batchSize * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

	float totalLoss = 0.0f;
	for (int i = 0; i < batchSize; i++) {
		if (isnan(h_losses[i]) || isinf(h_losses[i])) {
			printf("Dernières valeurs: pred=%.3f target=%.3f reg=%.3e\n",
				d_activations[vectOffsets[nbHiddenLayers + 1] + i * outputSize],
				d_targets[i * outputSize],
				regTotal);
			exit(EXIT_FAILURE);
		}
		totalLoss += h_losses[i];
	}

	return totalLoss / batchSize;
}

////////// Update Layer //////////

// Mise à jour des poids et biais avec ADAM pour chaque couche et chaque entrée du batch
__global__ void updateLayerADAM(
	int inSize, int outSize, int batchSize, float lr, float beta1, float beta2, float weightDecay, float epsilon, int t,
	float* weights, float* biases, float* activations, float* d_error,
	float* mWeights, float* vWeights, float* mBiases, float* vBiases
) {
	int column = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (row < inSize && column < outSize) {
		float dw = 0.f;
		float db = 0.f;

		// Calcul des gradients moyens pour le batch
		for (int i = 0; i < batchSize; i++) {
			float act = activations[i * inSize + row];
			float dl = d_error[i * outSize + column];
			dw += act * dl;
			db += dl;
		}

		// Moyenne des gradients pour ce batch
		dw /= batchSize;
		db /= batchSize;

		// Mise à jour des moments d'ordre 1 (m) et 2 (v) pour les poids
		int weightIdx = row * outSize + column;
		mWeights[weightIdx] = beta1 * mWeights[weightIdx] + (1 - beta1) * dw;
		vWeights[weightIdx] = beta2 * vWeights[weightIdx] + (1 - beta2) * (dw * dw);

		// Mise à jour des moments d'ordre 1 (m) et 2 (v) pour les biais
		mBiases[column] = beta1 * mBiases[column] + (1 - beta1) * db;
		vBiases[column] = beta2 * vBiases[column] + (1 - beta2) * (db * db);

		// Correction de biais pour les moments d'ordre 1 et 2
		float mWeightsHat = mWeights[weightIdx] / (1 - powf(beta1, t));
		float vWeightsHat = vWeights[weightIdx] / (1 - powf(beta2, t));

		float mBiasesHat = mBiases[column] / (1 - powf(beta1, t));
		float vBiasesHat = vBiases[column] / (1 - powf(beta2, t));

		// Mise à jour des poids et des biais selon ADAM
		weights[weightIdx] -= (lr * mWeightsHat) / (sqrtf(vWeightsHat) + epsilon) * weightDecay * weights[weightIdx];
		biases[column] -= (lr * mBiasesHat) / (sqrtf(vBiasesHat) + epsilon) * weightDecay * biases[column];
	}
}

void Network::updateWeightsNetwork(float learningRate) {
	const float beta1 = 0.9f;
	const float beta2 = 0.999f;
	const float epsilon = 1e-8f;

	for (int layer = 0; layer < nbHiddenLayers + 1; layer++) {
		int inSize = layerSizes[layer];
		int outSize = layerSizes[layer + 1];
		dim3 dimGrid((outSize - 1) / dimBlock.x + 1, (inSize - 1) / dimBlock.y + 1, 1);

		// Lancement du kernel pour la mise à jour de la couche actuelle
		updateLayerADAM << <dimGrid, dimBlock >> > (
			inSize,                             // Taille de la couche précédente (nombre d'entrées)
			outSize,                            // Taille de la couche actuelle (nombre de sorties)
			batchSize,                          // Taille du batch
			learningRate,                       // Taux d'apprentissage
			beta1,                              // Paramètre beta1 pour le moment d'ordre 1
			beta2,                              // Paramètre beta2 pour le moment d'ordre 2
			weightDecay,
			epsilon,                            // Épsilon pour la stabilité numérique
			t,                                  // Numéro de l'itération pour la correction de biais
			d_weights + matOffsets[layer],      // Poids de la couche actuelle
			d_biases + vectOffsets[layer],      // Biais de la couche actuelle
			d_activations + vectOffsets[layer], // Activations de la couche précédente
			d_errors + vectOffsets[layer],      // Erreurs de la couche actuelle
			d_mWeights + matOffsets[layer],     // Moments d'ordre 1 pour les poids de la couche
			d_vWeights + matOffsets[layer],     // Moments d'ordre 2 pour les poids de la couche
			d_mBiases + vectOffsets[layer],     // Moments d'ordre 1 pour les biais de la couche
			d_vBiases + vectOffsets[layer]);    // Moments d'ordre 2 pour les biais de la couche
		cudaDeviceSynchronize();
	}
	t++;
}

// Sauvegarde des poids
void Network::saveWeightsAndBiases(const std::string& filename) const {
	std::ofstream file(filename, std::ios::out | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Erreur : Impossible d'ouvrir le fichier pour la sauvegarde." << std::endl;
		return;
	}

	// Sauvegarder les poids et biais pour chaque couche
	for (int layer = 0; layer < nbHiddenLayers + 1; layer++) {
		int inputSize = layerSizes[layer];
		int outputSize = layerSizes[layer + 1];

		// Allocation d'un tampon temporaire sur l'hôte pour les poids et biais
		float* host_weights = new float[inputSize * outputSize];
		float* host_biases = new float[outputSize];

		// Copier les poids et biais du GPU vers le tampon hôte
		cudaMemcpy(host_weights, d_weights + matOffsets[layer],
			inputSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_biases, d_biases + vectOffsets[layer],
			outputSize * sizeof(float), cudaMemcpyDeviceToHost);

		// Sauvegarde des poids et biais depuis le tampon hôte vers le fichier
		file.write(reinterpret_cast<const char*>(host_weights), inputSize * outputSize * sizeof(float));
		file.write(reinterpret_cast<const char*>(host_biases), outputSize * sizeof(float));

		// Libération des tampons temporaires
		delete[] host_weights;
		delete[] host_biases;
	}

	file.close();
	std::cout << "Sauvegarde des poids et biais effectuée avec succès dans le fichier " << filename << std::endl;
}

// Chargement des poids
void Network::loadWeightsAndBiases(const std::string& filename) const {
	std::ifstream file(filename, std::ios::in | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Erreur : Impossible d'ouvrir le fichier pour charger les poids et biais." << std::endl;
		return;
	}

	// Charger les poids et biais pour chaque couche
	for (int layer = 0; layer < nbHiddenLayers + 1; layer++) {
		int inputSize = layerSizes[layer];
		int outputSize = layerSizes[layer + 1];

		// Allocation d'un tampon temporaire sur l'hôte pour les poids et biais
		float* host_weights = new float[inputSize * outputSize];
		float* host_biases = new float[outputSize];

		// Chargement des poids et biais dans le tampon temporaire
		file.read(reinterpret_cast<char*>(host_weights), inputSize * outputSize * sizeof(float));
		file.read(reinterpret_cast<char*>(host_biases), outputSize * sizeof(float));

		// Copier les poids et biais de l'hôte vers le GPU
		cudaMemcpy(d_weights + matOffsets[layer], host_weights,
			inputSize * outputSize * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_biases + vectOffsets[layer], host_biases,
			outputSize * sizeof(float), cudaMemcpyHostToDevice);

		// Libération des tampons temporaires
		delete[] host_weights;
		delete[] host_biases;
	}

	file.close();
	std::cout << "Chargement des poids et biais effectué avec succès depuis le fichier " << filename << std::endl;
}

// Opérateur pour copier seulement les poids et biais
Network& Network::operator=(const Network& other) {
	if (this != &other) {
		// Vérification de la compatibilité des réseaux
		if (this->inputSize != other.inputSize ||
			this->hiddenSize != other.hiddenSize ||
			this->outputSize != other.outputSize ||
			this->nbHiddenLayers != other.nbHiddenLayers) {
			throw std::invalid_argument("Incompatible network sizes for assignment.");
		}

		// Calcul de la taille totale des poids et biais
		int weightsSize = 0, biasesSize = 0;
		for (int layer = 0; layer < nbHiddenLayers + 1; layer++) {
			weightsSize += layerSizes[layer] * layerSizes[layer + 1];
			biasesSize += layerSizes[layer + 1];
		}

		// Libérer la mémoire existante
		cudaFree(d_weights);
		cudaFree(d_biases);

		// Allouer à nouveau la mémoire sur le GPU
		cudaMalloc(&d_weights, weightsSize * sizeof(float));
		cudaMalloc(&d_biases, biasesSize * sizeof(float));

		// Copier les données de `other` vers `this`
		cudaMemcpy(d_weights, other.d_weights, weightsSize * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaMemcpy(d_biases, other.d_biases, biasesSize * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	cudaDeviceSynchronize();
	return *this;
}
*/
