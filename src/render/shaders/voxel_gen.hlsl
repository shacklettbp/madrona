#include "shader_utils.hlsl"

[[vk::push_constant]]
VoxelGenPushConst pushConst;

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> vbo;

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> ibo;

[[vk::binding(2, 0)]]
StructuredBuffer<uint> voxels;


int coord(int x,int y,int z,int blockX,int blockY,int blockZ)
{
    return x * blockY * blockZ + y * blockZ + z;
}

#define xDim 64
#define yDim 1
#define zDim 1

[numThreads(xDim, yDim, zDim)]
[shader("compute")]
void voxelGen(uint3 idx : SV_DispatchThreadID)
{

	int indexVal = 0;
    float halfBlockSize = pushConst.blockWidth/2;
    float texAtlasStep = 1.0 / pushConst.numBlocks;

    int workPerThread = ceil(pushConst.worldX / (float)xDim);

    for(int i2=0;i2<workPerThread;i2++){
        int i = i2 + idx.x * workPerThread;
        if(i >= pushConst.worldX){
            return;
        }

        for (int j = 0; j < pushConst.worldY; j++) {
            for (int k = 0; k < pushConst.worldZ; k++) {
                uint data = voxels[coord(i,j,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ)];
                if (data != 0) {
                    float blockCenterX = halfBlockSize + pushConst.blockWidth * i;
                    float blockCenterY = halfBlockSize + pushConst.blockWidth * j;
                    float blockCenterZ = halfBlockSize + pushConst.blockWidth * k;

                    int index = 32*6*coord(i,j,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ);
                    int indexindex = 6*6*coord(i,j,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ);
                    int vertexIndex = coord(i,j,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ)*6*4;

                    int blockID = data - 1;

                    uint leftN = (i > 0) ? voxels[coord(i-1,j,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ)]: 0;
                    if (!leftN) {
                        //position
                        vbo[index + 0] = blockCenterX - halfBlockSize;
                        vbo[index + 1] = blockCenterY - halfBlockSize;
                        vbo[index + 2] = blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 3] = texAtlasStep * (blockID);
                        vbo[index + 4] = -0.333;

                        //normals
                        vbo[index+5] = -1;
                        vbo[index+6] = 0;
                        vbo[index+7] = 0;


                        //position
                        vbo[index + 8] = blockCenterX - halfBlockSize;
                        vbo[index + 9] = blockCenterY + halfBlockSize;
                        vbo[index + 10] =   blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 11] = texAtlasStep * (blockID + 1);
                        vbo[index + 12] = -0.333;

                        //normals
                        vbo[index+13] = -1;
                        vbo[index+14] = 0;
                        vbo[index+15] = 0;


                        //position
                        vbo[index + 16] = blockCenterX - halfBlockSize;
                        vbo[index + 17] = blockCenterY - halfBlockSize;
                        vbo[index + 18] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 19] = texAtlasStep * (blockID);
                        vbo[index + 20] = 0;

                        //normals
                        vbo[index+21] = -1;
                        vbo[index+22] = 0;
                        vbo[index+23] = 0;


                        //position
                        vbo[index + 24] = blockCenterX - halfBlockSize;
                        vbo[index + 25] = blockCenterY + halfBlockSize;
                        vbo[index + 26] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 27] = texAtlasStep * (blockID + 1);
                        vbo[index + 28] = 0;

                        //normals
                        vbo[index+29] = -1;
                        vbo[index+30] = 0;
                        vbo[index+31] = 0;
                    }

                    ibo[indexindex+0] = vertexIndex + 2;
                    ibo[indexindex+1] = vertexIndex + 1;
                    ibo[indexindex+2] = vertexIndex + 0;
                    ibo[indexindex+3] = vertexIndex + 3;
                    ibo[indexindex+4] = vertexIndex + 1;
                    ibo[indexindex+5] = vertexIndex + 2;


                    index+=32;
                    indexindex+=6;
                    vertexIndex+=4;

                    uint rightN = (i < pushConst.worldX - 1) ? voxels[coord(i+1,j,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ)] : 0;
                    if (!rightN) {
                        //position
                        vbo[index + 0] = blockCenterX + halfBlockSize;
                        vbo[index + 1] = blockCenterY - halfBlockSize;
                        vbo[index + 2] =  blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 3] = texAtlasStep * (blockID);
                        vbo[index + 4] = -0.333;

                        //normals
                        vbo[index+5] = 1;
                        vbo[index+6] = 0;
                        vbo[index+7] = 0;


                        //position
                        vbo[index + 8] = blockCenterX + halfBlockSize;
                        vbo[index + 9] = blockCenterY + halfBlockSize;
                        vbo[index + 10] =   blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 11] = texAtlasStep * (blockID + 1);
                        vbo[index + 12] = -0.333;

                        //normals
                        vbo[index+13] = 1;
                        vbo[index+14] = 0;
                        vbo[index+15] = 0;


                        //position
                        vbo[index + 16] = blockCenterX + halfBlockSize;
                        vbo[index + 17] = blockCenterY - halfBlockSize;
                        vbo[index + 18] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 19] = texAtlasStep * (blockID);
                        vbo[index + 20] = 0;

                        //normals
                        vbo[index+21] = 1;
                        vbo[index+22] = 0;
                        vbo[index+23] = 0;


                        //position
                        vbo[index + 24] = blockCenterX + halfBlockSize;
                        vbo[index + 25] = blockCenterY + halfBlockSize;
                        vbo[index + 26] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 27] = texAtlasStep * (blockID + 1);
                        vbo[index + 28] = 0;

                        //normals
                        vbo[index+29] = 1;
                        vbo[index+30] = 0;
                        vbo[index+31] = 0;
                    }

                    ibo[indexindex+0] = vertexIndex + 0;
                    ibo[indexindex+1] = vertexIndex + 1;
                    ibo[indexindex+2] = vertexIndex + 2;
                    ibo[indexindex+3] = vertexIndex + 2;
                    ibo[indexindex+4] = vertexIndex + 1;
                    ibo[indexindex+5] = vertexIndex + 3;

                    index+=32;
                    indexindex+=6;
                    vertexIndex+=4;

                    uint topN = (k < pushConst.worldZ - 1) ? voxels[coord(i,j,k+1,pushConst.worldX,pushConst.worldY,pushConst.worldZ)] : 0;
                    if (!topN) {
                        //position
                        vbo[index + 0] = (blockCenterX - halfBlockSize);
                        vbo[index + 1] = blockCenterY - halfBlockSize;
                        vbo[index + 2] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 3] = texAtlasStep * (blockID);
                        vbo[index + 4] = 0.333;

                        //normals
                        vbo[index+5] = 0;
                        vbo[index+6] = 0;
                        vbo[index+7] = 1;


                        //position
                        vbo[index + 8] = blockCenterX + halfBlockSize;
                        vbo[index + 9] = blockCenterY - halfBlockSize;
                        vbo[index + 10] = blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 11] = texAtlasStep * (blockID);
                        vbo[index + 12] = 0.667;

                        //normals
                        vbo[index+13] = 0;
                        vbo[index+14] = 0;
                        vbo[index+15] = 1;


                        //position
                        vbo[index + 16] = blockCenterX - halfBlockSize;
                        vbo[index + 17] = blockCenterY + halfBlockSize;
                        vbo[index + 18] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 19] = texAtlasStep * (blockID+1);
                        vbo[index + 20] = 0.333;

                        //normals
                        vbo[index+21] = 0;
                        vbo[index+22] = 0;
                        vbo[index+23] = 1;


                        //position
                        vbo[index + 24] = blockCenterX + halfBlockSize;
                        vbo[index + 25] = blockCenterY + halfBlockSize;
                        vbo[index + 26] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 27] = texAtlasStep * (blockID + 1);
                        vbo[index + 28] = 0.667;

                        //normals
                        vbo[index+29] = 0;
                        vbo[index+30] = 0;
                        vbo[index+31] = 1;
                    }

                    ibo[indexindex+0] = vertexIndex + 0;
                    ibo[indexindex+1] = vertexIndex + 1;
                    ibo[indexindex+2] = vertexIndex + 2;
                    ibo[indexindex+3] = vertexIndex + 2;
                    ibo[indexindex+4] = vertexIndex + 1;
                    ibo[indexindex+5] = vertexIndex + 3;

                    index+=32;
                    indexindex+=6;
                    vertexIndex+=4;

                    uint bottomN = (k > 0) ? voxels[coord(i,j,k-1,pushConst.worldX,pushConst.worldY,pushConst.worldZ)] : 0;
                    if (!bottomN) {
                        //position
                        vbo[index + 0] = blockCenterX - halfBlockSize;
                        vbo[index + 1] = blockCenterY - halfBlockSize;
                        vbo[index + 2] =  blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 3] = texAtlasStep * (blockID);
                        vbo[index + 4] = 0.667;

                        //normals
                        vbo[index+5] = 0;
                        vbo[index+6] = 0;
                        vbo[index+7] = -1;


                        //position
                        vbo[index + 8] = blockCenterX + halfBlockSize;
                        vbo[index + 9] = blockCenterY - halfBlockSize;
                        vbo[index + 10] =   blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 11] = texAtlasStep * (blockID);
                        vbo[index + 12] = 1;

                        //normals
                        vbo[index+13] = 0;
                        vbo[index+14] = 0;
                        vbo[index+15] = -1;


                        //position
                        vbo[index + 16] = blockCenterX - halfBlockSize;
                        vbo[index + 17] = blockCenterY + halfBlockSize;
                        vbo[index + 18] =  blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 19] = texAtlasStep * (blockID+1);
                        vbo[index + 20] = 0.667;

                        //normals
                        vbo[index+21] = 0;
                        vbo[index+22] = 0;
                        vbo[index+23] = -1;


                        //position
                        vbo[index + 24] = blockCenterX + halfBlockSize;
                        vbo[index + 25] = blockCenterY + halfBlockSize;
                        vbo[index + 26] =  blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 27] = texAtlasStep * (blockID + 1);
                        vbo[index + 28] = 1;

                        //normals
                        vbo[index+29] = 0;
                        vbo[index+30] = 0;
                        vbo[index+31] = -1;
                    }

                    ibo[indexindex+0] = vertexIndex + 2;
                    ibo[indexindex+1] = vertexIndex + 1;
                    ibo[indexindex+2] = vertexIndex + 0;
                    ibo[indexindex+3] = vertexIndex + 3;
                    ibo[indexindex+4] = vertexIndex + 1;
                    ibo[indexindex+5] = vertexIndex + 2;

                    index+=32;
                    indexindex+=6;
                    vertexIndex+=4;

                    uint frontN = (j < pushConst.worldY - 1) ? voxels[coord(i,j+1,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ)] : 0;
                    if (!frontN) {


                        //position
                        vbo[index + 0] = blockCenterX - halfBlockSize;
                        vbo[index + 1] = blockCenterY + halfBlockSize;
                        vbo[index + 2] =  blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 3] = texAtlasStep * (blockID);
                        vbo[index + 4] = -0.333;

                        //normals
                        vbo[index+5] = 0;
                        vbo[index+6] = 1;
                        vbo[index+7] = 0;


                        //position
                        vbo[index + 8] = blockCenterX + halfBlockSize;
                        vbo[index + 9] = blockCenterY + halfBlockSize;
                        vbo[index + 10] =   blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 11] = texAtlasStep * (blockID+1);
                        vbo[index + 12] = -0.333;

                        //normals
                        vbo[index+13] = 0;
                        vbo[index+14] = 1;
                        vbo[index+15] = 0;


                        //position
                        vbo[index + 16] = blockCenterX - halfBlockSize;
                        vbo[index + 17] = blockCenterY + halfBlockSize;
                        vbo[index + 18] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 19] = texAtlasStep * (blockID);
                        vbo[index + 20] = 0;

                        //normals
                        vbo[index+21] = 0;
                        vbo[index+22] = 1;
                        vbo[index+23] = 0;


                        //position
                        vbo[index + 24] = blockCenterX + halfBlockSize;
                        vbo[index + 25] = blockCenterY + halfBlockSize;
                        vbo[index + 26] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 27] = texAtlasStep * (blockID + 1);
                        vbo[index + 28] = 0;

                        //normals
                        vbo[index+29] = 0;
                        vbo[index+30] = 1;
                        vbo[index+31] = 0;
                    }

                    ibo[indexindex+0] = vertexIndex + 2;
                    ibo[indexindex+1] = vertexIndex + 1;
                    ibo[indexindex+2] = vertexIndex + 0;
                    ibo[indexindex+3] = vertexIndex + 3;
                    ibo[indexindex+4] = vertexIndex + 1;
                    ibo[indexindex+5] = vertexIndex + 2;

                    index+=32;
                    indexindex+=6;
                    vertexIndex+=4;

                    uint backN = (j > 0) ? voxels[coord(i,j-1,k,pushConst.worldX,pushConst.worldY,pushConst.worldZ)] : 0;
                    if (!backN) {


                        //position
                        vbo[index + 0] = blockCenterX - halfBlockSize;
                        vbo[index + 1] = blockCenterY - halfBlockSize;
                        vbo[index + 2] =  blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 3] = texAtlasStep * (blockID);
                        vbo[index + 4] = -0.333;

                        //normals
                        vbo[index+5] = 0;
                        vbo[index+6] = -1;
                        vbo[index+7] = 0;


                        //position
                        vbo[index + 8] = blockCenterX + halfBlockSize;
                        vbo[index + 9] = blockCenterY - halfBlockSize;
                        vbo[index + 10] =   blockCenterZ - halfBlockSize;

                        //tex coord
                        vbo[index + 11] = texAtlasStep * (blockID+1);
                        vbo[index + 12] = -0.333;

                        //normals
                        vbo[index+13] = 0;
                        vbo[index+14] = -1;
                        vbo[index+15] = 0;


                        //position
                        vbo[index + 16] = blockCenterX - halfBlockSize;
                        vbo[index + 17] = blockCenterY - halfBlockSize;
                        vbo[index + 18] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 19] = texAtlasStep * (blockID);
                        vbo[index + 20] = 0;

                        //normals
                        vbo[index+21] = 0;
                        vbo[index+22] = -1;
                        vbo[index+23] = 0;


                        //position
                        vbo[index + 24] = blockCenterX + halfBlockSize;
                        vbo[index + 25] = blockCenterY - halfBlockSize;
                        vbo[index + 26] =  blockCenterZ + halfBlockSize;

                        //tex coord
                        vbo[index + 27] = texAtlasStep * (blockID + 1);
                        vbo[index + 28] = 0;

                        //normals
                        vbo[index+29] = 0;
                        vbo[index+30] = -1;
                        vbo[index+31] = 0;

                    }

                    ibo[indexindex + 0] = vertexIndex + 0;
                    ibo[indexindex + 1] = vertexIndex + 1;
                    ibo[indexindex + 2] = vertexIndex + 2;
                    ibo[indexindex + 3] = vertexIndex + 2;
                    ibo[indexindex + 4] = vertexIndex + 1;
                    ibo[indexindex + 5] = vertexIndex + 3;

                }
            }
        }
    }
}
