use plonky2::hash::hash_types::RichField;
use plonky2::iop::ext_target::ExtensionTarget;
use plonky2::plonk::circuit_builder::CircuitBuilder;
use plonky2_field::extension::Extendable;
use plonky2_field::goldilocks_field::GoldilocksField;
use plonky2_field::types::Field;
use unroll::unroll_for_loops;

use crate::hash::poseidon2::{Poseidon2, SPONGE_WIDTH};

const MAT_INTERNAL_DIAG_12: [u64; 12] = [
    0xC3B6C08E23BA92FF, 0xD84B5DE94A324FB5, 0x0D0C371C5B35B84E, 0x7964F570E7188036,
    0x5DAF18BBD996604A, 0x6743BC47B9595256, 0x5528B9362C59BB6F, 0xAC45E25B7127B68A,
    0xA2077D7DFBB606B4, 0xF3FAAC6FAEE378AD, 0x0C6388B51545E882, 0xD27DBB6944917B5F,
];

const FULL_RC_12: [u64; 8 * 12] = [
    // R0
    0x13dcf33aba214f46, 0x30b3b654a1da6d83, 0x1fc634ada6159b56, 0x937459964dc03466,
    0xedd2ef2ca7949924, 0xede9affde0e22f68, 0x8515b9d6bac9282d, 0x6b5c07b4e9e900d8,
    0x1ec66368838c8a08, 0x9042367d80d1fbab, 0x400283564a3c3799, 0x4a00be0466bca75e,
    // R1
    0x7913beee58e3817f, 0xf545e88532237d90, 0x22f8cb8736042005, 0x6f04990e247a2623,
    0xfe22e87ba37c38cd, 0xd20e32c85ffe2815, 0x117227674048fe73, 0x4e9fb7ea98a6b145,
    0xe0866c232b8af08b, 0x00bbc77916884964, 0x7031c0fb990d7116, 0x240a9e87cf35108f,
    // R2
    0x2e6363a5a12244b3, 0x5e1c3787d1b5011c, 0x4132660e2a196e8b, 0x3a013b648d3d4327,
    0xf79839f49888ea43, 0xfe85658ebafe1439, 0xb6889825a14240bd, 0x578453605541382b,
    0x4508cda8f6b63ce9, 0x9c3ef35848684c91, 0x0812bde23c87178c, 0xfe49638f7f722c14,
    // R3
    0x8e3f688ce885cbf5, 0xb8e110acf746a87d, 0xb4b2e8973a6dabef, 0x9e714c5da3d462ec,
    0x6438f9033d3d0c15, 0x24312f7cf1a27199, 0x23f843bb47acbf71, 0x9183f11a34be9f01,
    0x839062fbb9d45dbf, 0x24b56e7e6c2e43fa, 0xe1683da61c962a72, 0xa95c63971a19bfa7,
    // R26
    0xc68be7c94882a24d, 0xaf996d5d5cdaedd9, 0x9717f025e7daf6a5, 0x6436679e6e7216f4,
    0x8a223d99047af267, 0xbb512e35a133ba9a, 0xfbbf44097671aa03, 0xf04058ebf6811e61,
    0x5cca84703fac7ffb, 0x9b55c7945de6469f, 0x8e05bf09808e934f, 0x2ea900de876307d7,
    // R27
    0x7748fff2b38dfb89, 0x6b99a676dd3b5d81, 0xac4bb7c627cf7c13, 0xadb6ebe5e9e2f5ba,
    0x2d33378cafa24ae3, 0x1e5b73807543f8c2, 0x09208814bfebb10f, 0x782e64b6bb5b93dd,
    0xadd5a48eac90b50f, 0xadd4c54c736ea4b1, 0xd58dbb86ed817fd8, 0x6d5ed1a533f34ddd,
    // R28
    0x28686aa3e36b7cb9, 0x591abd3476689f36, 0x047d766678f13875, 0xa2a11112625f5b49,
    0x21fd10a3f8304958, 0xf9b40711443b0280, 0xd2697eb8b2bde88e, 0x3493790b51731b3f,
    0x11caf9dd73764023, 0x7acfb8f72878164e, 0x744ec4db23cefc26, 0x1e00e58f422c6340,
    // R29
    0x21dd28d906a62dda, 0xf32a46ab5f465b5f, 0xbfce13201f3f7e6b, 0xf30d2e7adb5304e2,
    0xecdf4ee4abad48e9, 0xf94e82182d395019, 0x4ee52e3744d887c5, 0xa1341c7cac0083b2,
    0x2302fb26c30c834a, 0xaea3c587273bf7d3, 0xf798e24961823ec7, 0x962deba3e9a2cd94,
];

const PARTIAL_RC_12: [u64; 22] = [
    0x4adf842aa75d4316, 0xf8fbb871aa4ab4eb, 0x68e85b6eb2dd6aeb, 0x07a0b06b2d270380,
    0xd94e0228bd282de4, 0x8bdd91d3250c5278, 0x209c68b88bba778f, 0xb5e18cdab77f3877,
    0xb296a3e808da93fa, 0x8370ecbda11a327e, 0x3f9075283775dad8, 0xb78095bb23c6aa84,
    0x3f36b9fe72ad4e5f, 0x69bc96780b10b553, 0x3f1d341f2eb7b881, 0x4e939e9815838818,
    0xda366b3ae2a31604, 0xbc89db1e7287d509, 0x6102f411f9ef5659, 0x58725c5e7ac1f0ab,
    0x0df5856c798883e7, 0xf7bb62a8da4c961b,
];


impl Poseidon2 for GoldilocksField {
    const INTERNAL_MATRIX_DIAG: [u64; SPONGE_WIDTH] = MAT_INTERNAL_DIAG_12;
    const FULL_ROUND_CONSTANTS: [u64; 8 * SPONGE_WIDTH] = FULL_RC_12;
    const PARTIAL_ROUND_CONSTANTS: [u64; 22] = PARTIAL_RC_12;

    #[inline(always)]
    #[unroll_for_loops]
    fn external_linear_layer(state: &mut [Self; SPONGE_WIDTH]) {
        // Manually unrolled 4x4 Mix for 3 chunks (SPONGE_WIDTH=12)
        // Chunk 0
        {
            let s0 = state[0];
            let s1 = state[1];
            let s2 = state[2];
            let s3 = state[3];
            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = s1 + s1 + t1;
            let t3 = s3 + s3 + t0;
            // t4 = t1*4 + t3
            let t1_2 = t1 + t1;
            let t4 = t1_2 + t1_2 + t3;
            // t5 = t0*4 + t2
            let t0_2 = t0 + t0;
            let t5 = t0_2 + t0_2 + t2;
            let t6 = t3 + t5;
            let t7 = t2 + t4;
            state[0] = t6;
            state[1] = t5;
            state[2] = t7;
            state[3] = t4;
        }
        // Chunk 1
        {
            let s0 = state[4];
            let s1 = state[5];
            let s2 = state[6];
            let s3 = state[7];
            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = s1 + s1 + t1;
            let t3 = s3 + s3 + t0;
            let t1_2 = t1 + t1;
            let t4 = t1_2 + t1_2 + t3;
            let t0_2 = t0 + t0;
            let t5 = t0_2 + t0_2 + t2;
            let t6 = t3 + t5;
            let t7 = t2 + t4;
            state[4] = t6;
            state[5] = t5;
            state[6] = t7;
            state[7] = t4;
        }
        // Chunk 2
        {
            let s0 = state[8];
            let s1 = state[9];
            let s2 = state[10];
            let s3 = state[11];
            let t0 = s0 + s1;
            let t1 = s2 + s3;
            let t2 = s1 + s1 + t1;
            let t3 = s3 + s3 + t0;
            let t1_2 = t1 + t1;
            let t4 = t1_2 + t1_2 + t3;
            let t0_2 = t0 + t0;
            let t5 = t0_2 + t0_2 + t2;
            let t6 = t3 + t5;
            let t7 = t2 + t4;
            state[8] = t6;
            state[9] = t5;
            state[10] = t7;
            state[11] = t4;
        }

        // Column Sums
        let sum0 = state[0] + state[4] + state[8];
        let sum1 = state[1] + state[5] + state[9];
        let sum2 = state[2] + state[6] + state[10];
        let sum3 = state[3] + state[7] + state[11];

        state[0] += sum0;
        state[4] += sum0;
        state[8] += sum0;
        state[1] += sum1;
        state[5] += sum1;
        state[9] += sum1;
        state[2] += sum2;
        state[6] += sum2;
        state[10] += sum2;
        state[3] += sum3;
        state[7] += sum3;
        state[11] += sum3;
    }

    #[inline(always)]
    fn external_linear_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        // Chunk 0
        {
            let s0 = state[0];
            let s1 = state[1];
            let s2 = state[2];
            let s3 = state[3];
            let t0 = builder.add_extension(s0, s1);
            let t1 = builder.add_extension(s2, s3);
            let s1_2 = builder.add_extension(s1, s1);
            let t2 = builder.add_extension(s1_2, t1);
            let s3_2 = builder.add_extension(s3, s3);
            let t3 = builder.add_extension(s3_2, t0);

            let t1_2 = builder.add_extension(t1, t1);
            let t1_4 = builder.add_extension(t1_2, t1_2);
            let t4 = builder.add_extension(t1_4, t3);

            let t0_2 = builder.add_extension(t0, t0);
            let t0_4 = builder.add_extension(t0_2, t0_2);
            let t5 = builder.add_extension(t0_4, t2);

            let t6 = builder.add_extension(t3, t5);
            let t7 = builder.add_extension(t2, t4);
            state[0] = t6;
            state[1] = t5;
            state[2] = t7;
            state[3] = t4;
        }
        // Chunk 1
        {
            let s0 = state[4];
            let s1 = state[5];
            let s2 = state[6];
            let s3 = state[7];
            let t0 = builder.add_extension(s0, s1);
            let t1 = builder.add_extension(s2, s3);
            let s1_2 = builder.add_extension(s1, s1);
            let t2 = builder.add_extension(s1_2, t1);
            let s3_2 = builder.add_extension(s3, s3);
            let t3 = builder.add_extension(s3_2, t0);

            let t1_2 = builder.add_extension(t1, t1);
            let t1_4 = builder.add_extension(t1_2, t1_2);
            let t4 = builder.add_extension(t1_4, t3);

            let t0_2 = builder.add_extension(t0, t0);
            let t0_4 = builder.add_extension(t0_2, t0_2);
            let t5 = builder.add_extension(t0_4, t2);

            let t6 = builder.add_extension(t3, t5);
            let t7 = builder.add_extension(t2, t4);
            state[4] = t6;
            state[5] = t5;
            state[6] = t7;
            state[7] = t4;
        }
        // Chunk 2
        {
            let s0 = state[8];
            let s1 = state[9];
            let s2 = state[10];
            let s3 = state[11];
            let t0 = builder.add_extension(s0, s1);
            let t1 = builder.add_extension(s2, s3);
            let s1_2 = builder.add_extension(s1, s1);
            let t2 = builder.add_extension(s1_2, t1);
            let s3_2 = builder.add_extension(s3, s3);
            let t3 = builder.add_extension(s3_2, t0);

            let t1_2 = builder.add_extension(t1, t1);
            let t1_4 = builder.add_extension(t1_2, t1_2);
            let t4 = builder.add_extension(t1_4, t3);

            let t0_2 = builder.add_extension(t0, t0);
            let t0_4 = builder.add_extension(t0_2, t0_2);
            let t5 = builder.add_extension(t0_4, t2);

            let t6 = builder.add_extension(t3, t5);
            let t7 = builder.add_extension(t2, t4);
            state[8] = t6;
            state[9] = t5;
            state[10] = t7;
            state[11] = t4;
        }

        let sum0 = builder.add_many_extension([state[0], state[4], state[8]]);
        let sum1 = builder.add_many_extension([state[1], state[5], state[9]]);
        let sum2 = builder.add_many_extension([state[2], state[6], state[10]]);
        let sum3 = builder.add_many_extension([state[3], state[7], state[11]]);
        let sums = [sum0, sum1, sum2, sum3];

        for i in 0..12 {
            state[i] = builder.add_extension(state[i], sums[i % 4]);
        }
    }

    #[inline(always)]
    fn internal_linear_layer(state: &mut [Self; SPONGE_WIDTH]) {
        let mut sum = Self::ZERO;
        for x in state.iter() {
            sum += *x;
        }
        for i in 0..SPONGE_WIDTH {
            state[i] = state[i] * Self::from_canonical_u64(Self::INTERNAL_MATRIX_DIAG[i]) + sum;
        }
    }

    fn internal_linear_layer_circuit<const D: usize>(
        builder: &mut CircuitBuilder<Self, D>,
        state: &mut [ExtensionTarget<D>; SPONGE_WIDTH],
    ) where
        Self: RichField + Extendable<D>,
    {
        let sum = builder.add_many_extension(*state);
        for i in 0..SPONGE_WIDTH {
            let diag = <Self as Extendable<D>>::Extension::from_canonical_u64(
                Self::INTERNAL_MATRIX_DIAG[i],
            );
            let diag_t = builder.constant_extension(diag);
            let term = builder.mul_extension(state[i], diag_t);
            state[i] = builder.add_extension(term, sum);
        }
    }
}

#[cfg(test)]
mod tests {
    use plonky2::field::goldilocks_field::GoldilocksField;
    use plonky2::plonk::config::Hasher;

    use super::*;
    use crate::hash::poseidon2::Poseidon2Hash;

    fn ensure_hash_match<const N: usize>(input: &[u64; N], output: &[u64; 4]) {
        let input_felts: [GoldilocksField; N] =
            core::array::from_fn(|i| GoldilocksField::from_canonical_u64(input[i]));
        let expected_output_felts: [GoldilocksField; 4] =
            core::array::from_fn(|i| GoldilocksField::from_canonical_u64(output[i]));
        let native_output = Poseidon2Hash::hash_no_pad(&input_felts);
        assert_eq!(native_output.elements, expected_output_felts, "Native hash output does not match expected output (input: {:?}, expected_output: {:?}, implementation output: {:?})", input, expected_output_felts, native_output.elements);
    }

    #[test]
    fn test_poseidon2_hash() {
        ensure_hash_match(&[], &[0, 0, 0, 0]);
        ensure_hash_match(&[], &[0, 0, 0, 0]);
        ensure_hash_match(
            &[0u64],
            &[
                2706484646582314364,
                16460758560799937193,
                2052063466144512209,
                9649607828149110866,
            ],
        );
        ensure_hash_match(
            &[2569661146879475556u64],
            &[
                16769639327691431973,
                7141594277929940324,
                9485945817594321380,
                11842582573248139679,
            ],
        );
        ensure_hash_match(
            &[0u64, 0u64],
            &[
                2706484646582314364,
                16460758560799937193,
                2052063466144512209,
                9649607828149110866,
            ],
        );
        ensure_hash_match(
            &[13598113309785430975u64, 9716569787960217193u64],
            &[
                6682130850969365932,
                2019040415149856338,
                16669705077741594128,
                1041576905718618790,
            ],
        );
        ensure_hash_match(
            &[15563908632086336797u64, 11939853829661265652u64],
            &[
                6754788449120798716,
                11537756638690742428,
                17278236322000039539,
                9446169535267893089,
            ],
        );
        ensure_hash_match(
            &[
                7035870691246949695u64,
                11766413715335812481u64,
                15263657140681447212u64,
            ],
            &[
                3281768623057315869,
                6104152705653136741,
                18290992612250282247,
                2291594416235545565,
            ],
        );
        ensure_hash_match(
            &[
                11086826210154843960u64,
                12932153918390625031u64,
                8603826665515618675u64,
                10228142193639134053u64,
            ],
            &[
                5383171767994636990,
                14917773042587379481,
                1491959057325711953,
                4457334068364519336,
            ],
        );
        ensure_hash_match(
            &[
                1085455116594784350u64,
                3137863944689450333u64,
                2827213793526951737u64,
                11482885912170411125u64,
                16396774079730511279u64,
            ],
            &[
                1773143387914224369,
                9538236312147012774,
                11649517970147396269,
                8648728750153090287,
            ],
        );
        ensure_hash_match(
            &[
                6132399403762204582u64,
                3417950990053826304u64,
                14215535460888266765u64,
                11940818035081356429u64,
                5885809234771532806u64,
                11612260045944963025u64,
            ],
            &[
                9622208190932827270,
                3282495381645058480,
                12530380485944186020,
                4950300771043791131,
            ],
        );
        ensure_hash_match(
            &[
                6916345689477195540u64,
                10773426897048722312u64,
                12078405691899781577u64,
                16650955282777129777u64,
                4947609528135747230u64,
                16600148043774216618u64,
                17343359682336418714u64,
            ],
            &[
                16927034253639706565,
                15334080234674377786,
                5951935220573325330,
                10824626484423512073,
            ],
        );
        ensure_hash_match(
            &[
                13347032926777936530u64,
                3570742615514676466u64,
                6878463522444605645u64,
                17626290524020802693u64,
                7670827384508807303u64,
                2119559300248034536u64,
                10390469946183390648u64,
            ],
            &[
                7920869566101566666,
                10530024564531307950,
                15569480194662356099,
                18028384291739899609,
            ],
        );
        ensure_hash_match(
            &[0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64],
            &[
                2706484646582314364,
                16460758560799937193,
                2052063466144512209,
                9649607828149110866,
            ],
        );
        ensure_hash_match(
            &[
                18187535067709718001u64,
                14337705152392438101u64,
                6982379327474982550u64,
                11859584835221828636u64,
                8249995159544805868u64,
                8035559166740697418u64,
                12528977149568582807u64,
                3427379066287458904u64,
            ],
            &[
                13432558199665100202,
                962163451901398214,
                4878222294242780339,
                5607479075057137573,
            ],
        );
        ensure_hash_match(
            &[
                10748408770276200846u64,
                4065885288698410909u64,
                6646184245268861050u64,
                17034995564664163169u64,
                11626180379727114422u64,
                17221464286381784878u64,
                930458622099283249u64,
                12905342626764675100u64,
            ],
            &[
                6415738861496453202,
                1873413804988806652,
                5202604532815810046,
                2338680893983848616,
            ],
        );
        ensure_hash_match(
            &[
                1801274305228244300u64,
                3524517047489834204u64,
                5040580846628039201u64,
                1833129323954120488u64,
                8361302219606021381u64,
                14331665959696843429u64,
                3189133317540083292u64,
                16211095615010480906u64,
                7560010853129571836u64,
            ],
            &[
                5921128472866251709,
                7890237649036567577,
                8590979135679112710,
                6743202784677583985,
            ],
        );
        ensure_hash_match(
            &[
                18119149162384539819u64,
                8535118413767625964u64,
                13770910845186340622u64,
                12210740442970958691u64,
                10284101585315034559u64,
                5412977715305543295u64,
                17547128477802331507u64,
                7105673766001812050u64,
                12701387576422406941u64,
            ],
            &[
                494891174441074650,
                13912591327536453587,
                13857820756302749482,
                11368881732732806660,
            ],
        );
        ensure_hash_match(
            &[
                0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64, 0u64,
            ],
            &[
                2196388115497925456,
                10258741142686394406,
                7890033718352538478,
                7842350413430956945,
            ],
        );
        ensure_hash_match(
            &[
                6713494575396728310u64,
                4498434768187162922u64,
                17187951315638283016u64,
                10084642066725624527u64,
                7487341086116173495u64,
                16157103071565312019u64,
                3371539043314892347u64,
                8142008541130077558u64,
                306051673530216135u64,
                2713080146132812134u64,
                17686241339677239133u64,
                16368878142543567290u64,
                5709786955392648410u64,
            ],
            &[
                15178103398577291480,
                7279733064200989166,
                7352247359037878949,
                6163290185652814691,
            ],
        );
        ensure_hash_match(
            &[
                13880209818773612783u64,
                1142249444678072965u64,
                13436581560857182469u64,
                9370472206854336294u64,
                12753436453101617005u64,
                4385458149260623640u64,
                2487740732562298238u64,
                11648546162156281441u64,
                3440684410853319814u64,
                3639110535962188619u64,
                15616833138943306818u64,
                16936656388580774504u64,
                2542340302183601468u64,
            ],
            &[
                8428037479880997412,
                2728168643836565168,
                11399559083354648808,
                15296410950943018322,
            ],
        );
        ensure_hash_match(
            &[
                9642835411944603919u64,
                7105291494275419671u64,
                10176809369188545914u64,
                7970153414652538956u64,
                13583820464557294711u64,
                43298137719606936u64,
                9934729252589019115u64,
                18091402841602700618u64,
                8983295197231930714u64,
                6130207910083869627u64,
                15898698217283859718u64,
                5376771619223607088u64,
                16668399049463529649u64,
            ],
            &[
                3480115054426605230,
                9215537843496030786,
                5250997139695738744,
                11585105258758447211,
            ],
        );
        ensure_hash_match(
            &[
                16679137655979289766u64,
                1615758668386064707u64,
                14474640405557776511u64,
                14371725890985354788u64,
                12192024729846328573u64,
                12511064215255473116u64,
                9886083264998037662u64,
                12999838127338264543u64,
                9342254808633019043u64,
                7092203266026798331u64,
                11700353157027101949u64,
                7352333675806324723u64,
                13002122565302142499u64,
                15378998605040838969u64,
                3384162765167147038u64,
                4594502360264837197u64,
            ],
            &[
                9934002541491996403,
                6153636309396364023,
                2663356342027927619,
                16245898949911246667,
            ],
        );
        ensure_hash_match(
            &[
                12364953552313713780u64,
                18122907158955934776u64,
                7086904763704665321u64,
                2847814009566578864u64,
                13730418109486949099u64,
                10393679557316046213u64,
                6052667481059866817u64,
                8207716471246086104u64,
                5836262252210395155u64,
                17793858611266222491u64,
                9561834685295833221u64,
                4851994297119806074u64,
                2153601147279080132u64,
                7875641775842186746u64,
                837548013736101859u64,
                895587739933738372u64,
                13576568093129166799u64,
            ],
            &[
                3853453024313313240,
                2578885930525834247,
                14287892012740467115,
                14953611030725437478,
            ],
        );
        ensure_hash_match(
            &[
                10200987805734511580u64,
                10852524306521761117u64,
                5445246291708628076u64,
                17237460148233781426u64,
                2931846680385680826u64,
                3790811905905907970u64,
                8633887072542628943u64,
                7351962051935983920u64,
                15864589543418773385u64,
                5237229298904735923u64,
                18436941025273921894u64,
                16625277622125521739u64,
                5955117935056879693u64,
                281324445306645224u64,
                12368762587926684947u64,
                14421207555087604266u64,
                12233958433524666615u64,
                10123886682953531720u64,
            ],
            &[
                10869717420438392251,
                12151701766075398998,
                1656758219965192320,
                4813847500521427709,
            ],
        );
    }
}
