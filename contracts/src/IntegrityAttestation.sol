// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {AccessControl} from "@openzeppelin/contracts/access/AccessControl.sol";

/**
 * @title IntegrityAttestation
 * @notice Stores zkML proof attestations and wallet classifications on-chain
 * @dev Uses OpenZeppelin AccessControl for role-based authorization.
 *      DEFAULT_ADMIN_ROLE manages roles. ATTESTER_ROLE can submit proofs
 *      and record classifications.
 */
contract IntegrityAttestation is AccessControl {
    // ============ Roles ============

    bytes32 public constant ATTESTER_ROLE = keccak256("ATTESTER_ROLE");

    // ============ Types ============

    enum Classification {
        GenuineCommerce,   // 0
        LowActivity,       // 1
        ScriptedBenign,    // 2
        CircularPayments,  // 3
        WashTrading        // 4
    }

    struct WalletRecord {
        Classification classification;
        uint8 confidence;      // 0-100
        bytes32 proofHash;
        uint256 timestamp;
        bool exists;
    }

    // ============ State Variables ============

    /// @notice Mapping of valid proof hashes
    mapping(bytes32 => bool) public validProofs;

    /// @notice Mapping of proof hash to submission timestamp
    mapping(bytes32 => uint256) public proofTimestamps;

    /// @notice Mapping of wallet address to latest classification
    mapping(address => WalletRecord) public walletRecords;

    /// @notice Total number of attested proofs
    uint256 public totalProofs;

    /// @notice Total number of classified wallets
    uint256 public totalWallets;

    // ============ Events ============

    event ProofAttested(
        bytes32 indexed proofHash,
        address indexed attester,
        uint256 timestamp
    );

    event ClassificationRecorded(
        address indexed wallet,
        Classification classification,
        uint8 confidence,
        bytes32 proofHash,
        uint256 timestamp
    );

    // ============ Errors ============

    error ProofAlreadyAttested();
    error InvalidProofHash();
    error InvalidConfidence();

    // ============ Constructor ============

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(ATTESTER_ROLE, msg.sender);
    }

    // ============ External Functions ============

    /**
     * @notice Submit a proof attestation
     * @param proofHash The keccak256 hash of the zkML proof
     */
    function attestProof(bytes32 proofHash) external onlyRole(ATTESTER_ROLE) {
        if (proofHash == bytes32(0)) revert InvalidProofHash();
        if (validProofs[proofHash]) revert ProofAlreadyAttested();

        validProofs[proofHash] = true;
        proofTimestamps[proofHash] = block.timestamp;
        totalProofs++;

        emit ProofAttested(proofHash, msg.sender, block.timestamp);
    }

    /**
     * @notice Record a wallet classification with proof
     * @param wallet The wallet address being classified
     * @param classification The classification result
     * @param confidence Confidence score (0-100)
     * @param proofHash The proof hash for this classification
     */
    function recordClassification(
        address wallet,
        Classification classification,
        uint8 confidence,
        bytes32 proofHash
    ) external onlyRole(ATTESTER_ROLE) {
        if (proofHash == bytes32(0)) revert InvalidProofHash();
        if (confidence > 100) revert InvalidConfidence();

        // Attest the proof if not already attested
        if (!validProofs[proofHash]) {
            validProofs[proofHash] = true;
            proofTimestamps[proofHash] = block.timestamp;
            totalProofs++;
            emit ProofAttested(proofHash, msg.sender, block.timestamp);
        }

        // Update wallet record
        bool isNew = !walletRecords[wallet].exists;
        walletRecords[wallet] = WalletRecord({
            classification: classification,
            confidence: confidence,
            proofHash: proofHash,
            timestamp: block.timestamp,
            exists: true
        });

        if (isNew) {
            totalWallets++;
        }

        emit ClassificationRecorded(
            wallet,
            classification,
            confidence,
            proofHash,
            block.timestamp
        );
    }

    /**
     * @notice Check if a proof hash is valid (attested)
     * @param proofHash The proof hash to check
     * @return True if the proof has been attested
     */
    function isProofHashValid(bytes32 proofHash) external view returns (bool) {
        return validProofs[proofHash];
    }

    /**
     * @notice Get the latest classification for a wallet
     * @param wallet The wallet address to query
     * @return classification The classification result
     * @return confidence The confidence score
     * @return proofHash The associated proof hash
     * @return timestamp When the classification was recorded
     */
    function getWalletClassification(address wallet)
        external
        view
        returns (
            Classification classification,
            uint8 confidence,
            bytes32 proofHash,
            uint256 timestamp
        )
    {
        WalletRecord memory record = walletRecords[wallet];
        return (
            record.classification,
            record.confidence,
            record.proofHash,
            record.timestamp
        );
    }

    /**
     * @notice Check if a wallet is flagged as suspicious
     * @param wallet The wallet address to check
     * @return True if classified as CircularPayments or WashTrading
     */
    function isWalletSuspicious(address wallet) external view returns (bool) {
        WalletRecord memory record = walletRecords[wallet];
        if (!record.exists) return false;
        return record.classification == Classification.CircularPayments
            || record.classification == Classification.WashTrading;
    }

    /**
     * @notice Check if an address is an authorized attester
     * @param attester The address to check
     * @return True if the address has ATTESTER_ROLE
     */
    function isAttester(address attester) external view returns (bool) {
        return hasRole(ATTESTER_ROLE, attester);
    }
}
