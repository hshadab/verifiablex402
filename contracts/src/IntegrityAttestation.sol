// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title IntegrityAttestation
 * @notice Stores zkML proof attestations and wallet classifications on-chain
 * @dev Proof hashes are submitted by authorized attesters. Stores both proof
 *      validity and the latest integrity classification per wallet.
 */
contract IntegrityAttestation {
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

    /// @notice Owner of the contract
    address public owner;

    /// @notice Mapping of valid proof hashes
    mapping(bytes32 => bool) public validProofs;

    /// @notice Mapping of proof hash to submission timestamp
    mapping(bytes32 => uint256) public proofTimestamps;

    /// @notice Mapping of authorized attesters
    mapping(address => bool) public attesters;

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

    event AttesterAdded(address indexed attester);
    event AttesterRemoved(address indexed attester);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    // ============ Errors ============

    error NotAuthorized();
    error NotOwner();
    error ProofAlreadyAttested();
    error InvalidProofHash();
    error InvalidConfidence();

    // ============ Constructor ============

    constructor() {
        owner = msg.sender;
        attesters[msg.sender] = true;
        emit AttesterAdded(msg.sender);
    }

    // ============ Modifiers ============

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    modifier onlyAttester() {
        if (!attesters[msg.sender] && msg.sender != owner) {
            revert NotAuthorized();
        }
        _;
    }

    // ============ External Functions ============

    /**
     * @notice Submit a proof attestation
     * @param proofHash The keccak256 hash of the zkML proof
     */
    function attestProof(bytes32 proofHash) external onlyAttester {
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
    ) external onlyAttester {
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
     * @notice Add an authorized attester
     * @param attester The address to authorize
     */
    function addAttester(address attester) external onlyOwner {
        attesters[attester] = true;
        emit AttesterAdded(attester);
    }

    /**
     * @notice Remove an authorized attester
     * @param attester The address to remove
     */
    function removeAttester(address attester) external onlyOwner {
        attesters[attester] = false;
        emit AttesterRemoved(attester);
    }

    /**
     * @notice Transfer ownership
     * @param newOwner The new owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        address previousOwner = owner;
        owner = newOwner;
        emit OwnershipTransferred(previousOwner, newOwner);
    }

    /**
     * @notice Check if an address is an authorized attester
     * @param attester The address to check
     * @return True if the address is authorized
     */
    function isAttester(address attester) external view returns (bool) {
        return attesters[attester] || attester == owner;
    }
}
