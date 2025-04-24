using UnityEngine;
using UnityEngine.Tilemaps;
using System.Collections.Generic;

public enum PentominoPiece
{
    F,
    I,
    L,
    N,
    P,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z
}

[System.Serializable]
public struct PentominoData
{
    public PentominoPiece pentomino;
    public Tile tile;
    public Vector2Int[] cells;

}