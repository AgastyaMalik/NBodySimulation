#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>

#define CHECK_CUDA_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

struct Body {
    double mass;
    double posx, posy, posz;
    double velx, vely, velz;
};

__global__ void computeForces(Body* bodies, double* fx, double* fy, double* fz, int n) {
    extern __shared__ Body shared_bodies[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int tid = threadIdx.x;
    double fx_i = 0.0, fy_i = 0.0, fz_i = 0.0;
    const double G = 6.67430e-11;
    const double softening = 1e-9;

    for (int tile = 0; tile < (n + blockDim.x - 1) / blockDim.x; ++tile) {
        int j = tile * blockDim.x + tid;
        if (j < n) {
            shared_bodies[tid] = bodies[j];
        }
        __syncthreads();

        int tile_size = min(blockDim.x, n - tile * blockDim.x);
        for (int k = 0; k < tile_size; ++k) {
            if (i == tile * blockDim.x + k) continue;
            double dx = shared_bodies[k].posx - bodies[i].posx;
            double dy = shared_bodies[k].posy - bodies[i].posy;
            double dz = shared_bodies[k].posz - bodies[i].posz;
            double r_squared = dx * dx + dy * dy + dz * dz + softening;
            double r = sqrt(r_squared);
            double force_magnitude = (G * bodies[i].mass * shared_bodies[k].mass) / r_squared;

            fx_i += force_magnitude * (dx / r);
            fy_i += force_magnitude * (dy / r);
            fz_i += force_magnitude * (dz / r);
        }
        __syncthreads();
    }

    fx[i] = fx_i;
    fy[i] = fy_i;
    fz[i] = fz_i;
}

__global__ void updateVelocities(Body* bodies, double* fx, double* fy, double* fz, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double ax = fx[i] / bodies[i].mass;
    double ay = fy[i] / bodies[i].mass;
    double az = fz[i] / bodies[i].mass;
    bodies[i].velx += ax * (dt / 2.0);
    bodies[i].vely += ay * (dt / 2.0);
    bodies[i].velz += az * (dt / 2.0);
}

__global__ void updatePositions(Body* bodies, int n, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    bodies[i].posx += bodies[i].velx * dt;
    bodies[i].posy += bodies[i].vely * dt;
    bodies[i].posz += bodies[i].velz * dt;
}

int main() {
    int n;
    std::cout << "Enter number of bodies to simulate: ";
    std::cin >> n;

    std::vector<Body> bodies(n);
    std::vector<Body> temp_bodies(n);
    for (int i = 0; i < n; ++i) {
        std::cout << "Enter details for body " << i + 1 << ":" << std::endl;
        std::cout << "Mass: ";
        std::cin >> bodies[i].mass;
        std::cout << "Position (x y z): ";
        std::cin >> bodies[i].posx >> bodies[i].posy >> bodies[i].posz;
        std::cout << "Velocity (vx vy vz): ";
        std::cin >> bodies[i].velx >> bodies[i].vely >> bodies[i].velz;
    }

    double dt;
    int timeSteps;
    std::cout << "Enter time step size (dt): ";
    std::cin >> dt;
    std::cout << "Enter number of time steps to simulate: ";
    std::cin >> timeSteps;

    // Open file for writing positions
    std::ofstream outFile("positions.txt");
    if (!outFile.is_open()) {
        std::cerr << "Error opening positions.txt" << std::endl;
        return 1;
    }

    Body* d_bodies;
    double *d_fx, *d_fy, *d_fz;
    CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, n * sizeof(Body)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fx, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fy, n * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_fz, n * sizeof(double)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(Body);

    for (int t = 0; t < timeSteps; ++t) {
        computeForces<<<blocks, threadsPerBlock, sharedMemSize>>>(d_bodies, d_fx, d_fy, d_fz, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        updateVelocities<<<blocks, threadsPerBlock>>>(d_bodies, d_fx, d_fy, d_fz, n, dt);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        updatePositions<<<blocks, threadsPerBlock>>>(d_bodies, n, dt);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        computeForces<<<blocks, threadsPerBlock, sharedMemSize>>>(d_bodies, d_fx, d_fy, d_fz, n);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        updateVelocities<<<blocks, threadsPerBlock>>>(d_bodies, d_fx, d_fy, d_fz, n, dt);
        CHECK_CUDA_ERROR(cudaGetLastError());
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        // Save positions to file
        CHECK_CUDA_ERROR(cudaMemcpy(temp_bodies.data(), d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost));
        outFile << "Time step " << t + 1 << "\n";
        for (int i = 0; i < n; ++i) {
            outFile << temp_bodies[i].posx << " " << temp_bodies[i].posy << " " << temp_bodies[i].posz << "\n";
        }
        outFile << "\n";

        // Print center of mass
        double cm_x = 0.0, cm_y = 0.0, cm_z = 0.0, total_mass = 0.0;
        for (int i = 0; i < n; ++i) {
            cm_x += temp_bodies[i].mass * temp_bodies[i].posx;
            cm_y += temp_bodies[i].mass * temp_bodies[i].posy;
            cm_z += temp_bodies[i].mass * temp_bodies[i].posz;
            total_mass += temp_bodies[i].mass;
        }
        cm_x /= total_mass; cm_y /= total_mass; cm_z /= total_mass;
        std::cout << "Time step " << t + 1 << " Center of mass: (" << cm_x << ", " << cm_y << ", " << cm_z << ")" << std::endl;
    }

    CHECK_CUDA_ERROR(cudaMemcpy(bodies.data(), d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost));

    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    CHECK_CUDA_ERROR(cudaFree(d_fx));
    CHECK_CUDA_ERROR(cudaFree(d_fy));
    CHECK_CUDA_ERROR(cudaFree(d_fz));

    std::cout << "Final positions:" << std::endl;
    for (int i = 0; i < n; ++i) {
        std::cout << "Body " << i + 1 << " position: (" << bodies[i].posx << ", " << bodies[i].posy << ", " << bodies[i].posz << ")" << std::endl;
    }

    outFile.close();
    return 0;
}   