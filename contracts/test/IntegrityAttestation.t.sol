// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import {Test} from "forge-std/Test.sol";
import {IntegrityAttestation} from "../src/IntegrityAttestation.sol";
import {IAccessControl} from "@openzeppelin/contracts/access/IAccessControl.sol";

contract IntegrityAttestationTest is Test {
    IntegrityAttestation public attestation;
    address public deployer;
    address public attester;
    address public unauthorized;
    address public wallet1;
    address public wallet2;

    bytes32 constant ATTESTER_ROLE = keccak256("ATTESTER_ROLE");
    bytes32 constant DEFAULT_ADMIN_ROLE = 0x00;

    function setUp() public {
        deployer = address(this);
        attester = makeAddr("attester");
        unauthorized = makeAddr("unauthorized");
        wallet1 = makeAddr("wallet1");
        wallet2 = makeAddr("wallet2");

        attestation = new IntegrityAttestation();
    }

    // ============ Deployment ============

    function test_DeployerIsAdmin() public view {
        assertTrue(attestation.hasRole(DEFAULT_ADMIN_ROLE, deployer));
    }

    function test_DeployerIsAttester() public view {
        assertTrue(attestation.isAttester(deployer));
    }

    // ============ Add / Remove Attester ============

    function test_AddAttester() public {
        attestation.grantRole(ATTESTER_ROLE, attester);
        assertTrue(attestation.isAttester(attester));
    }

    function test_RemoveAttester() public {
        attestation.grantRole(ATTESTER_ROLE, attester);
        assertTrue(attestation.isAttester(attester));

        attestation.revokeRole(ATTESTER_ROLE, attester);
        assertFalse(attestation.isAttester(attester));
    }

    function test_UnauthorizedCannotGrantRole() public {
        vm.prank(unauthorized);
        vm.expectRevert(
            abi.encodeWithSelector(
                IAccessControl.AccessControlUnauthorizedAccount.selector,
                unauthorized,
                DEFAULT_ADMIN_ROLE
            )
        );
        attestation.grantRole(ATTESTER_ROLE, unauthorized);
    }

    // ============ Attest Proof ============

    function test_AttestProof() public {
        bytes32 proofHash = keccak256("proof1");
        attestation.attestProof(proofHash);

        assertTrue(attestation.isProofHashValid(proofHash));
        assertEq(attestation.totalProofs(), 1);
        assertGt(attestation.proofTimestamps(proofHash), 0);
    }

    function test_AttestProofDuplicate() public {
        bytes32 proofHash = keccak256("proof1");
        attestation.attestProof(proofHash);

        vm.expectRevert(IntegrityAttestation.ProofAlreadyAttested.selector);
        attestation.attestProof(proofHash);
    }

    function test_AttestProofZeroHash() public {
        vm.expectRevert(IntegrityAttestation.InvalidProofHash.selector);
        attestation.attestProof(bytes32(0));
    }

    function test_UnauthorizedCannotAttest() public {
        vm.prank(unauthorized);
        vm.expectRevert(
            abi.encodeWithSelector(
                IAccessControl.AccessControlUnauthorizedAccount.selector,
                unauthorized,
                ATTESTER_ROLE
            )
        );
        attestation.attestProof(keccak256("proof"));
    }

    // ============ Record Classification ============

    function test_RecordClassification() public {
        bytes32 proofHash = keccak256("proof1");

        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.GenuineCommerce,
            85,
            proofHash
        );

        (
            IntegrityAttestation.Classification cls,
            uint8 confidence,
            bytes32 proof,
            uint256 timestamp
        ) = attestation.getWalletClassification(wallet1);

        assertEq(uint8(cls), uint8(IntegrityAttestation.Classification.GenuineCommerce));
        assertEq(confidence, 85);
        assertEq(proof, proofHash);
        assertGt(timestamp, 0);
        assertEq(attestation.totalWallets(), 1);
        // Proof should also be attested
        assertTrue(attestation.isProofHashValid(proofHash));
    }

    function test_RecordClassificationInvalidConfidence() public {
        vm.expectRevert(IntegrityAttestation.InvalidConfidence.selector);
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.GenuineCommerce,
            101,
            keccak256("proof")
        );
    }

    function test_RecordClassificationZeroHash() public {
        vm.expectRevert(IntegrityAttestation.InvalidProofHash.selector);
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.GenuineCommerce,
            50,
            bytes32(0)
        );
    }

    function test_UnauthorizedCannotRecordClassification() public {
        vm.prank(unauthorized);
        vm.expectRevert(
            abi.encodeWithSelector(
                IAccessControl.AccessControlUnauthorizedAccount.selector,
                unauthorized,
                ATTESTER_ROLE
            )
        );
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.GenuineCommerce,
            50,
            keccak256("proof")
        );
    }

    // ============ isWalletSuspicious ============

    function test_IsWalletSuspicious_CircularPayments() public {
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.CircularPayments,
            70,
            keccak256("proof1")
        );
        assertTrue(attestation.isWalletSuspicious(wallet1));
    }

    function test_IsWalletSuspicious_WashTrading() public {
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.WashTrading,
            90,
            keccak256("proof2")
        );
        assertTrue(attestation.isWalletSuspicious(wallet1));
    }

    function test_NotSuspicious_GenuineCommerce() public {
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.GenuineCommerce,
            80,
            keccak256("proof3")
        );
        assertFalse(attestation.isWalletSuspicious(wallet1));
    }

    function test_NotSuspicious_Unclassified() public view {
        assertFalse(attestation.isWalletSuspicious(wallet1));
    }

    // ============ Counter Increments ============

    function test_CounterIncrements() public {
        assertEq(attestation.totalProofs(), 0);
        assertEq(attestation.totalWallets(), 0);

        attestation.attestProof(keccak256("p1"));
        assertEq(attestation.totalProofs(), 1);

        attestation.attestProof(keccak256("p2"));
        assertEq(attestation.totalProofs(), 2);

        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.LowActivity,
            50,
            keccak256("p3")
        );
        assertEq(attestation.totalProofs(), 3);
        assertEq(attestation.totalWallets(), 1);

        // Second classification for same wallet should not increment totalWallets
        attestation.recordClassification(
            wallet1,
            IntegrityAttestation.Classification.GenuineCommerce,
            60,
            keccak256("p4")
        );
        assertEq(attestation.totalWallets(), 1);
        assertEq(attestation.totalProofs(), 4);

        // New wallet increments
        attestation.recordClassification(
            wallet2,
            IntegrityAttestation.Classification.ScriptedBenign,
            75,
            keccak256("p5")
        );
        assertEq(attestation.totalWallets(), 2);
    }

    // ============ Role Transfer ============

    function test_AdminCanTransferAdminRole() public {
        address newAdmin = makeAddr("newAdmin");
        attestation.grantRole(DEFAULT_ADMIN_ROLE, newAdmin);
        assertTrue(attestation.hasRole(DEFAULT_ADMIN_ROLE, newAdmin));

        // New admin can grant attester role
        vm.prank(newAdmin);
        attestation.grantRole(ATTESTER_ROLE, makeAddr("newAttester"));
    }

    // ============ Attester via grantRole ============

    function test_GrantedAttesterCanAttest() public {
        attestation.grantRole(ATTESTER_ROLE, attester);

        vm.prank(attester);
        attestation.attestProof(keccak256("attester_proof"));
        assertTrue(attestation.isProofHashValid(keccak256("attester_proof")));
    }
}
